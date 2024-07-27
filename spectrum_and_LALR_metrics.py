import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import json
from prompt_toolkit.shortcuts import checkboxlist_dialog, input_dialog
import argparse
from tqdm import tqdm
import os
import time

class ModelModifier:
    def __init__(self, model_name=None, top_percent=50, bottom_percent=None, batch_size=1):
        self.model_name = model_name
        self.top_percent = top_percent
        self.bottom_percent = bottom_percent
        self.batch_size = batch_size

        if model_name:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32, 
                    low_cpu_mem_usage=True, 
                    trust_remote_code=True, 
                    device_map="auto"
                )
            except KeyError as e:
                print(f"Error loading model: {e}")
                print("Attempting to load with custom configuration...")
                config = AutoConfig.from_pretrained(model_name)
                config.rope_scaling = {"type": "linear", "factor": 1.0}
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=config,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map="auto"
                )

            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True, add_prefix_space=True)
            
            if not hasattr(self.model.config, 'rope_scaling'):
                self.model.config.rope_scaling = {'type': 'linear'}
            elif not isinstance(self.model.config.rope_scaling, dict):
                self.model.config.rope_scaling = {'type': 'linear'}
            elif 'type' not in self.model.config.rope_scaling:
                self.model.config.rope_scaling['type'] = 'linear'
        else:
            self.model = None
            self.optimizer = None
            self.tokenizer = None

        self.layer_snr = {}
        self.layer_svd = {}
        self.layer_ce = {}
        self.layer_cs = {}
        self.layer_types = []

    def get_weight_types(self):
        weight_types = set()
        for name, module in self.model.named_modules():
            parts = name.split('.')
            if any(hasattr(module, attr) for attr in ['weight', 'bias','inv_freq']):
                layer_index = next((i for i, part in enumerate(parts) if part.isdigit()), -1)
                weight_type = '.'.join(parts[layer_index + 1:]) if layer_index != -1 else name
                weight_types.add(weight_type)
        return list(weight_types)

    def interactive_select_weights(self):
        weight_types = self.get_weight_types()
        sorted_weight_types = self.sort_weight_types(weight_types)
        selected_types = checkboxlist_dialog(
            title="Select Weight Types", 
            text="Deselect the weight types you do not want to scan for metrics:",
            values=[(wt, wt) for wt in sorted_weight_types],
            default_values=sorted_weight_types
        ).run()
        self.layer_types = selected_types
        return selected_types

    def sort_weight_types(self, weight_types):
        categories = {}
        for wt in weight_types:
            category = wt.split('.')[0]
            categories.setdefault(category, []).append(wt)
        sorted_categories = {k: sorted(v) for k, v in sorted(categories.items(), key=lambda item: item[0])}
        sorted_weight_types = [wt for sublist in sorted_categories.values() for wt in sublist]
        return sorted_weight_types

    def assess_layers_metrics(self, selected_weight_types):
        start_time = time.time()

        for layer_type in tqdm(selected_weight_types, desc='Calculating metrics for types'):
            layers = [(name, module) for name, module in self.model.named_modules() if layer_type in name and hasattr(module, 'weight')]
            if not layers:
                continue

            all_weights = []
            layer_names = []

            for name, module in layers:
                weights = module.weight.detach()
                if weights.ndim < 2:
                    weights = weights.unsqueeze(0)
                all_weights.append(weights)
                layer_names.append(name)

            # Calculate metrics for all layers of this type
            snr_metrics = self.calculate_snr_metric(all_weights)
            svd_metrics = self.calculate_svd_metric(all_weights)
            ce_metrics = self.calculate_ce_metric(all_weights)
            cs_metrics = self.calculate_cs_metric(all_weights)

            # Normalize metrics within this layer type (except SNR)
            svd_metrics = self.normalize_metric(svd_metrics)
            ce_metrics = self.normalize_metric(ce_metrics)
            cs_metrics = self.normalize_metric(cs_metrics)

            # Store metrics
            for name, snr, svd, ce, cs in zip(layer_names, snr_metrics, svd_metrics, ce_metrics, cs_metrics):
                self.layer_snr[name] = {'type': layer_type, 'snr': snr}
                self.layer_svd[name] = {'type': layer_type, 'svd': svd}
                self.layer_ce[name] = {'type': layer_type, 'ce': ce}
                self.layer_cs[name] = {'type': layer_type, 'cs': cs}

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    def calculate_snr_metric(self, weights_list):
        snr_metrics = []
        for weights in weights_list:
            S = torch.linalg.svdvals(weights)
            max_singular_value = S[0]
            sigma_estimated = self.estimate_sigma_with_full_iqr(S)
            n, m = weights.shape[-2:]
            mp_threshold = self.marchenko_pastur_threshold(sigma_estimated, n, m)
            signal = S[S > mp_threshold].sum()
            noise = S[S <= mp_threshold].sum()
            snr = signal / noise if noise != 0 else float('inf')
            snr_metrics.append(snr / max_singular_value)
        return snr_metrics

    def calculate_svd_metric(self, weights_list):
        svd_metrics = []
        for weights in weights_list:
            S = torch.linalg.svdvals(weights)
            svd_metric = 1 - (torch.mean(S) / S[0])  # 1 - (Mean singular value normalized by largest singular value)
            svd_metrics.append(svd_metric.item())
        return svd_metrics

    def calculate_ce_metric(self, weights_list):
        ce_metrics = []
        all_weights = torch.cat([w.flatten() for w in weights_list])
        global_p = torch.softmax(all_weights, dim=0)
        max_ce = -torch.sum(global_p * torch.log(global_p + 1e-10))  # Maximum possible CE
        for weights in weights_list:
            p = torch.softmax(weights.flatten(), dim=0)
            ce = -torch.sum(p * torch.log(p + 1e-10))
            normalized_ce = ce / max_ce  # Normalize to [0, 1]
            ce_metrics.append(1 - normalized_ce.item())  # 1 - normalized CE so higher values mean higher importance
        return ce_metrics

    def calculate_cs_metric(self, weights_list):
        cs_metrics = []
        all_weights = torch.cat([w.flatten() for w in weights_list])
        all_weights_norm = all_weights / torch.norm(all_weights)
        for weights in weights_list:
            weights_norm = weights.flatten() / torch.norm(weights)
            cs = torch.dot(weights_norm, all_weights_norm)
            cs_metrics.append(cs.item())
        return cs_metrics

    @staticmethod
    def normalize_metric(metric_list):
        metric_array = np.array(metric_list)
        min_val, max_val = np.min(metric_array), np.max(metric_array)
        if min_val == max_val:
            return np.ones_like(metric_array)
        return (metric_array - min_val) / (max_val - min_val)

    @staticmethod
    def marchenko_pastur_threshold(sigma, n, m):
        beta = n / m if n < m else m / n
        threshold = sigma * np.sqrt((1 + np.sqrt(beta)) ** 2)
        return threshold

    @staticmethod
    def estimate_sigma_with_full_iqr(S):
        q75 = torch.quantile(S, 0.75)
        q25 = torch.quantile(S, 0.25)
        iqr = q75 - q25
        sigma_estimated = iqr / 1.349
        return sigma_estimated

    def save_metric_to_json(self, metric_name, metric_data):
        model_name_slug = self.model_name.replace('/', '-').replace('_', '-')
        directory = 'model_metrics_results'
        filename = os.path.join(directory, f'{metric_name}_results_{model_name_slug}.json')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        serializable_data = {}
        for layer_name, info in metric_data.items():
            metric_value = info[metric_name].item() if isinstance(info[metric_name], torch.Tensor) else info[metric_name]
            layer_type = str(info['type'])
            serializable_data[layer_name] = {metric_name: metric_value, 'type': layer_type}
        
        with open(filename, 'w') as file:
            json.dump(serializable_data, file, indent=4)
        
        print(f"{metric_name.upper()} results saved to {filename}")

    def save_all_metrics_to_json(self):
        self.save_metric_to_json('snr', self.layer_snr)
        self.save_metric_to_json('svd', self.layer_svd)
        self.save_metric_to_json('ce', self.layer_ce)
        self.save_metric_to_json('cs', self.layer_cs)

    def generate_unfrozen_params_yaml(self, json_filename, top_percent=None, bottom_percent=None, metric='snr'):
        top_percent = top_percent if top_percent is not None else self.top_percent
        bottom_percent = bottom_percent if bottom_percent is not None else self.bottom_percent
        
        if top_percent is not None and bottom_percent is not None:
            raise ValueError("Cannot specify both top_percent and bottom_percent")
        
        metric = metric.lower()
        if metric not in ['snr', 'svd', 'ce', 'cs']:
            raise ValueError(f"Invalid metric: {metric}. Choose from 'snr', 'svd', 'ce', or 'cs'.")
        
        metric_data = getattr(self, f'layer_{metric}')
        
        unfrozen_parameters = {}
        for layer_name, info in metric_data.items():
            layer_type = info['type']
            if layer_type not in unfrozen_parameters:
                unfrozen_parameters[layer_type] = []
            unfrozen_parameters[layer_type].append((layer_name, info[metric]))
        
        top_layers_by_type = {}
        for layer_type, layers in unfrozen_parameters.items():
            layers_sorted = sorted(layers, key=lambda x: x[1], reverse=True)
            if top_percent is not None:
                num_layers = int(len(layers) * top_percent / 100)
                selected_layers = layers_sorted[:num_layers]
            elif bottom_percent is not None:
                num_layers = int(len(layers) * bottom_percent / 100)
                selected_layers = layers_sorted[-num_layers:]
            else:
                selected_layers = layers_sorted
            top_layers_by_type[layer_type] = [layer[0] for layer in selected_layers]
        
        json_file_base = os.path.splitext(os.path.basename(json_filename))[0]
        percent_type = "top" if top_percent is not None else "bottom"
        percent_value = top_percent if top_percent is not None else bottom_percent
        yaml_filename = f"{json_file_base}_unfrozenparameters_{metric}_{percent_type}{percent_value}percent.yaml"
        
        with open(yaml_filename, 'w') as file:
            file.write("unfrozen_parameters:\n")
            file.write("- ^lm_head.weight$\n")
            file.write("- ^model.embed_tokens.weight$\n")
            for layer_type, layer_names in top_layers_by_type.items():
                file.write(f"# {layer_type} layers\n")
                for layer_name in layer_names:
                    file.write(f"- {layer_name}\n")
        print(f"{percent_type.capitalize()} {percent_value}% {metric.upper()} layers saved to {yaml_filename}")

    def generate_lr_scalars_yaml(self, metric='snr'):
        metric = metric.lower()
        if metric not in ['snr', 'svd', 'ce', 'cs']:
            raise ValueError(f"Invalid metric: {metric}. Choose from 'snr', 'svd', 'ce', or 'cs'.")
        
        metric_data = getattr(self, f'layer_{metric}')
        
        model_name_slug = self.model_name.replace('/', '-').replace('_', '-')
        yaml_filename = f"lr_scalars_{metric}_{model_name_slug}.yaml"
        
        # Normalize SNR values if SNR is the chosen metric
        if metric == 'snr':
            snr_values = [info['snr'] for info in metric_data.values()]
            normalized_snr = self.normalize_metric(snr_values)
            normalized_snr_dict = dict(zip(metric_data.keys(), normalized_snr))
        
        with open(yaml_filename, 'w') as file:
            file.write("lr_scalars:\n")
            for layer_name, info in metric_data.items():
                if metric == 'snr':
                    scalar = normalized_snr_dict[layer_name]
                else:
                    scalar = info[metric]
                file.write(f"  {layer_name}: {scalar:.6f}\n")
        
        print(f"Learning rate scalars based on {metric.upper()} saved to {yaml_filename}")

# Handle command-line arguments
parser = argparse.ArgumentParser(description="Process metrics data for layers.")
parser.add_argument('--model-name', type=str, required=True, help='Model name or path to the model')
parser.add_argument('--top-percent', type=int, default=None, help='Top percentage of layers to select, overriding the default')
parser.add_argument('--bottom-percent', type=int, default=None, help='Bottom percentage of layers to select, overriding the default')
parser.add_argument('--metric', type=str, default='snr', choices=['snr', 'svd', 'ce', 'cs'],
                    help='Select either SNR, SVD, CE, or CS metric for .yaml unfrozen parameter output. Default is SNR.')
args = parser.parse_args()

if args.top_percent is not None and args.bottom_percent is not None:
    raise ValueError("Cannot specify both --top-percent and --bottom-percent")

# Check for existing SNR results file
model_name_slug = args.model_name.replace('/', '-').replace('_', '-')
snr_file_path = os.path.join('model_metrics_results', f'snr_results_{model_name_slug}.json')

if os.path.exists(snr_file_path):
    print(f"Found existing SNR results file for {args.model_name}")
    modifier = ModelModifier(model_name=args.model_name, top_percent=args.top_percent, bottom_percent=args.bottom_percent)
    modifier.generate_unfrozen_params_yaml(snr_file_path, args.top_percent, args.bottom_percent, args.metric)
    modifier.generate_lr_scalars_yaml(args.metric)
else:
    print(f"No existing SNR results file found for {args.model_name}. Proceeding with metrics calculation.")
    batch_size = input_dialog(title="Batch Size", text="Enter the batch size:").run()
    batch_size = int(batch_size) if batch_size else 1
    modifier = ModelModifier(model_name=args.model_name, batch_size=batch_size, top_percent=args.top_percent, bottom_percent=args.bottom_percent)
    selected_weight_types = modifier.interactive_select_weights()
    if selected_weight_types:
        modifier.assess_layers_metrics(selected_weight_types)
        modifier.save_all_metrics_to_json()
        modifier.generate_unfrozen_params_yaml(snr_file_path, args.top_percent, args.bottom_percent, args.metric)
        modifier.generate_lr_scalars_yaml(args.metric)
        print("Finished metrics scanning and data saved.")
    else:
        print("No weight types selected.")