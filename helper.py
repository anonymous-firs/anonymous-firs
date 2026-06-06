from shutil import copyfile

import math
import torch

from torch.autograd import Variable
import logging
import sklearn.metrics.pairwise as smp
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import time

logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record

class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name

        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
        self.fg= FoolsGold(use_memory=self.params['fg_use_memory'])


        # === Experiment bookkeeping (overhead / data-efficiency) ===
        self._metrics_jsonl = os.path.join(self.folder_path, self.params.get('metrics_jsonl', 'metrics_round.jsonl'))
        self.reset_round_stats()
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm(model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_max_values(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer.data - target_params[name].data)))
        return squared_sum

    @staticmethod
    def model_max_values_var(model, target_params):
        squared_sum = list()
        for name, layer in model.named_parameters():
            squared_sum.append(torch.max(torch.abs(layer - target_params[name])))
        return sum(squared_sum)

    @staticmethod
    def get_one_vec(model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

        return sum_var

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.FloatTensor(size).fill_(0)
        sum_var= sum_var.to(config.device)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def cos_sim_loss(self, model, target_vec):
        model_vec = self.get_one_vec(model, variable=True)
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(
            self.params['scale_weights'] * (model_vec - target_var) + target_var, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        logger.info("los")
        logger.info(cs_sim.data[0])
        logger.info(torch.norm(model_vec - target_var).data[0])
        loss = 1 - cs_sim

        return 1e3 * loss

    def model_cosine_similarity(self, model, target_params_variables,
                                model_id='attacker'):

        cs_list = list()
        cs_loss = torch.nn.CosineSimilarity(dim=0)
        for name, data in model.named_parameters():
            if name == 'decoder.weight':
                continue

            model_update = 100 * (data.view(-1) - target_params_variables[name].view(-1)) + target_params_variables[
                name].view(-1)

            cs = F.cosine_similarity(model_update,
                                     target_params_variables[name].view(-1), dim=0)
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        logger.info(model_id)
        logger.info((sum(cs_list) / len(cs_list)).data[0])
        return 1e3 * sum(cos_los_submit)

    def accum_similarity(self, last_acc, new_acc):

        cs_list = list()

        cs_loss = torch.nn.CosineSimilarity(dim=0)
        # logger.info('new run')
        for name, layer in last_acc.items():
            cs = cs_loss(Variable(last_acc[name], requires_grad=False).view(-1),
                         Variable(new_acc[name], requires_grad=False).view(-1))
            # logger.info(torch.equal(layer.view(-1),
            #                          target_params_variables[name].view(-1)))
            # logger.info(name)
            # logger.info(cs.data[0])
            # logger.info(torch.norm(model_update).data[0])
            # logger.info(torch.norm(fake_weights[name]))
            cs_list.append(cs)
        cos_los_submit = 1 * (1 - sum(cs_list) / len(cs_list))
        # logger.info("AAAAAAAA")
        # logger.info((sum(cs_list)/len(cs_list)).data[0])
        return sum(cos_los_submit)

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.empty_like(param, dtype=torch.float32, device=param.device).normal_(mean=0.0, std=sigma)
        return noised_layer

    def accumulate_weight(self, weight_accumulator, epochs_submit_update_dict, state_keys, num_samples_dict):
        if self.params['aggregation_methods'] == config.AGGR_FOOLSGOLD:
            updates = dict()
            for i in range(len(state_keys)):
                local_model_gradients = epochs_submit_update_dict[state_keys[i]][0]
                num_samples = num_samples_dict[state_keys[i]]
                updates[state_keys[i]] = (num_samples, copy.deepcopy(local_model_gradients))
            return None, updates
        else:
            updates = dict()
            for i in range(len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                update = dict()
                num_samples = num_samples_dict[state_keys[i]]


                for name, data in local_model_update_list[0].items():
                    if torch.is_floating_point(data):
                        update[name] = torch.zeros_like(data, dtype=data.dtype, device=data.device)

                for j in range(len(local_model_update_list)):
                    local_model_update_dict = local_model_update_list[j]
                    for name, data in local_model_update_dict.items():
                        if name not in weight_accumulator:
                            continue
                        if not torch.is_tensor(data):
                            continue

                        upd = data.to(dtype=weight_accumulator[name].dtype, device=weight_accumulator[name].device)
                        weight_accumulator[name].add_(upd)
                        update[name].add_(upd)


                        detached_data = data.detach().cpu().numpy().tolist()
                        local_model_update_dict[name] = detached_data

                updates[state_keys[i]] = (num_samples, update)

            return weight_accumulator, updates

    def init_weight_accumulator(self, target_model):
        weight_accumulator = dict()
        for name, data in target_model.state_dict().items():
            if torch.is_floating_point(data):

                weight_accumulator[name] = torch.zeros_like(data, dtype=torch.float32, device=data.device)
        return weight_accumulator

    def average_shrink_models(self, weight_accumulator, target_model, epoch_interval):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue
            if not torch.is_floating_point(data):
                continue
            upd = weight_accumulator.get(name)
            if upd is None:
                continue

            update_per_layer = upd * (self.params["eta"] / self.params["no_models"])
            # update_per_layer = upd * (self.params["eta"] / self.params["number_of_total_participants"])
            # update_per_layer = update_per_layer * 1.0 / epoch_interval

            if self.params['diff_privacy']:
                noise = self.dp_noise(data, self.params['sigma'])
                update_per_layer = update_per_layer.to(dtype=noise.dtype, device=noise.device)
                update_per_layer = update_per_layer + noise

            data.add_(update_per_layer.to(dtype=data.dtype, device=data.device))
        return True


    def foolsgold_update(self,target_model,updates):
        client_grads = []
        alphas = []
        names = []
        for name, data in updates.items():
            client_grads.append(data[1])  # gradient
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[foolsgold agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[foolsgold agg] considering poison per batch poison_fraction: {poison_fraction}')

        target_model.train()
        # train and update
        optimizer = torch.optim.SGD(target_model.parameters(), lr=self.params['lr'],
                                    momentum=self.params['momentum'],
                                    weight_decay=self.params['decay'])

        optimizer.zero_grad()
        agg_grads, wv,alpha = self.fg.aggregate_gradients(client_grads,names)
        for i, (name, params) in enumerate(target_model.named_parameters()):
            agg_grads[i]=agg_grads[i] * self.params["eta"]
            if params.requires_grad:
                params.grad = agg_grads[i].to(config.device)
        optimizer.step()
        wv=wv.tolist()
        utils.csv_record.add_weight_result(names, wv, alpha)
        return True, names, wv, alpha

    def geometric_median_update(self, target_model, updates, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm=None):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm"""
        points = []
        alphas = []
        names = []
        for name, data in updates.items():
            points.append(data[1])  # update
            alphas.append(data[0])  # num_samples
            names.append(name)

        adver_ratio = 0
        for i in range(0, len(names)):
            _name = names[i]
            if _name in self.params['adversary_list']:
                adver_ratio += alphas[i]
        adver_ratio = adver_ratio / sum(alphas)
        poison_fraction = adver_ratio * self.params['poisoning_per_batch'] / self.params['batch_size']
        logger.info(f'[rfa agg] training data poison_ratio: {adver_ratio}  data num: {alphas}')
        logger.info(f'[rfa agg] considering poison per batch poison_fraction: {poison_fraction}')

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        alphas = torch.from_numpy(alphas).float()

        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        wv = None
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                dtype=alphas.dtype)
            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                        (prev_obj_val - obj_val) / obj_val,
                        Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                wv = copy.deepcopy(weights)
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: {abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv = copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0.0
        for name, data in median.items():
            if torch.is_floating_point(data):
                update_norm += torch.sum(torch.pow(data, 2))
        update_norm = math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                if not torch.is_floating_point(data):
                    continue
                if name not in median:
                    continue
                update_per_layer = median[name] * (self.params["eta"])
                if self.params['diff_privacy']:
                    noise = self.dp_noise(data, self.params['sigma'])
                    update_per_layer = update_per_layer.to(dtype=noise.dtype, device=noise.device)
                    update_per_layer = update_per_layer + noise
                data.add_(update_per_layer.to(dtype=data.dtype, device=data.device))
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)
        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(), alphas

    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

        # return sum([alpha * Helper.l2dist(median, p) for alpha, p in zip(alphas, points)])

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()


        for name, data in points[0].items():
            if torch.is_floating_point(data):
                weighted_updates[name] = torch.zeros_like(data, dtype=torch.float32, device=data.device)
        for w, p in zip(weights, points):
            for name, acc in weighted_updates.items():

                tmp = (w / tot_weights).to(acc.dtype)
                contrib = (p[name].to(acc.dtype) * tmp).to(device=acc.device)
                acc.add_(contrib)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

    def update_epoch_submit_dict(self,epochs_submit_update_dict,global_epochs_submit_dict, epoch,state_keys):

        epoch_len= len(epochs_submit_update_dict[state_keys[0]])
        for j in range(0, epoch_len):
            per_epoch_dict = dict()
            for i in range(0, len(state_keys)):
                local_model_update_list = epochs_submit_update_dict[state_keys[i]]
                local_model_update_dict = local_model_update_list[j]
                per_epoch_dict[state_keys[i]]= local_model_update_dict

            global_epochs_submit_dict[epoch+j]= per_epoch_dict

        return global_epochs_submit_dict


    def save_epoch_submit_dict(self, global_epochs_submit_dict):
        with open(f'{self.folder_path}/epoch_submit_update.json', 'w') as outfile:
            json.dump(global_epochs_submit_dict, outfile, ensure_ascii=False, indent=1)

    def estimate_fisher(self, model, criterion,
                        data_loader, sample_size, batch_size=64):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        if self.params['type'] == 'text':
            data_iterator = range(0, data_loader.size(0) - 1, self.params['bptt'])
            hidden = model.init_hidden(self.params['batch_size'])
        else:
            data_iterator = data_loader

        for batch_id, batch in enumerate(data_iterator):
            data, targets = self.get_batch(data_loader, batch,
                                           evaluation=False)
            if self.params['type'] == 'text':
                hidden = self.repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                loss = criterion(output.view(-1, self.n_tokens), targets)
            else:
                output = model(data)
                loss = log_softmax(output, dim=1)[range(targets.shape[0]), targets.data]
                # loss = criterion(output.view(-1, ntokens
            # output, hidden = model(data, hidden)
            loglikelihoods.append(loss)
            # loglikelihoods.append(
            #     log_softmax(output.view(-1, self.n_tokens))[range(self.params['batch_size']), targets.data]
            # )

            # if len(loglikelihoods) >= sample_size // batch_size:
            #     break
        logger.info(loglikelihoods[0].shape)
        # estimate the fisher information of the parameters.
        loglikelihood = torch.cat(loglikelihoods).mean(0)
        logger.info(loglikelihood.shape)
        loglikelihood_grads = torch.autograd.grad(loglikelihood, model.parameters())

        parameter_names = [
            n.replace('.', '__') for n, p in model.named_parameters()
        ]
        return {n: g ** 2 for n, g in zip(parameter_names, loglikelihood_grads)}

    def consolidate(self, model, fisher):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            model.register_buffer('{}_estimated_mean'.format(n), p.data.clone())
            model.register_buffer('{}_estimated_fisher'
                                  .format(n), fisher[n].data.clone())

    def ewc_loss(self, model, lamda, cuda=False):
        try:
            losses = []
            for n, p in model.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace('.', '__')
                mean = getattr(model, '{}_estimated_mean'.format(n))
                fisher = getattr(model, '{}_estimated_fisher'.format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

    # =========================
    # FLTrust implementation
    # =========================
    def _norm_of_update(self, upd: dict) -> torch.Tensor:
        """L2 norm over all floating point tensors in an update dict."""
        s = torch.tensor(0.0, dtype=torch.float32, device=next(self.target_model.parameters()).device if self.target_model is not None else 'cpu')
        for name, v in upd.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                s = s + torch.sum(v.float() * v.float())
        return torch.sqrt(s + 1e-12)

    def _apply_aggregated_update(self, target_model, agg_update: dict, eta: float) -> bool:
        """Apply aggregated update dict to target_model in-place with step size eta."""
        try:
            for name, p in target_model.named_parameters():
                if name in agg_update and torch.is_floating_point(p):
                    upd = agg_update[name].to(dtype=p.dtype, device=p.device)
                    p.data.add_(upd * float(eta))
            return True
        except Exception as e:
            logger.exception(f"[fltrust] apply update failed: {e}")
            return False

    def build_fltrust_root_loader(self):
        """
        Build the FLTrust root DataLoader.
        If params['fltrust_root_dir'] exists, use it as an ImageFolder root.
        Otherwise, sample a small class-balanced clean root set from train_dataset.
        """
        import os, random, collections
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms

        bs      = int(self.params.get('fltrust_batch_size', 64))
        size    = int(self.params.get('fltrust_root_size', 1000))
        workers = int(self.params.get('fltrust_root_workers', 0))
        pin     = bool(self.params.get('fltrust_root_pin_memory', False))
        rootdir = self.params.get('fltrust_root_dir', None)


        if rootdir and isinstance(rootdir, str) and os.path.isdir(rootdir):
            # Reuse an available training transform; fall back to ToTensor().
            tfm = None
            try:
                td = self.train_data
                sample_loader = None
                if isinstance(td, list) and len(td) > 0:
                    sample_loader = td[0][1] if isinstance(td[0], (list, tuple)) and len(td[0]) >= 2 else td[0]
                elif isinstance(td, dict) and len(td) > 0:
                    any_v = next(iter(td.values()))
                    sample_loader = any_v[1] if isinstance(any_v, (list, tuple)) and len(any_v) >= 2 else any_v
                elif td is not None:
                    sample_loader = td
                if hasattr(sample_loader, "dataset"):
                    tfm = getattr(sample_loader.dataset, "transform", None)
            except Exception:
                tfm = None
            if tfm is None:
                tfm = transforms.ToTensor()

            ds = datasets.ImageFolder(rootdir, transform=tfm)
            dl = DataLoader(ds, batch_size=bs, shuffle=True,
                            num_workers=workers, pin_memory=pin)
            logger.info(f"[FLTrust] root(dir): size={len(ds)}, bs={bs}, workers={workers}, pin={pin}")
            return dl


        ds = getattr(self, "train_dataset", None)
        if ds is None:
            raise RuntimeError("[FLTrust] train_dataset is None; cannot build root loader")

        N = len(ds)

        if hasattr(ds, "targets"):
            targets = list(ds.targets)
        elif hasattr(ds, "samples"):
            targets = [cls for (_p, cls) in ds.samples]
        else:

            idx = torch.randperm(N)[:size].tolist()
            sub = Subset(ds, idx)
            dl  = DataLoader(sub, batch_size=bs, shuffle=True,
                            num_workers=workers, pin_memory=pin)
            logger.info(f"[FLTrust] root(sampled): size={len(idx)}/{N}, bs={bs}, workers={workers}, pin={pin}")
            return dl


        cls_to_idx = collections.defaultdict(list)
        for i, y in enumerate(targets):
            cls_to_idx[int(y)].append(i)

        num_classes = len(cls_to_idx)
        per_class   = max(1, size // max(1, num_classes))

        chosen = []
        for _, idxs in cls_to_idx.items():
            random.shuffle(idxs)
            take = min(per_class, len(idxs))
            chosen.extend(idxs[:take])


        if len(chosen) < size:
            rest = list(set(range(N)) - set(chosen))
            random.shuffle(rest)
            chosen += rest[:(size - len(chosen))]

        random.shuffle(chosen)
        sub = Subset(ds, chosen)
        dl  = DataLoader(sub, batch_size=bs, shuffle=True,
                        num_workers=workers, pin_memory=pin)
        logger.info(f"[FLTrust] root(sampled, class-balanced): size={len(chosen)}/{N}, bs={bs}, workers={workers}, pin={pin}")
        return dl

    def _vectorize_update(self, upd: dict) -> torch.Tensor:
        """Flatten update dict to a single vector (float32, device of model)."""
        device = next(self.target_model.parameters()).device
        vecs = []
        for name, p in self.target_model.named_parameters():
            if name in upd and torch.is_floating_point(upd[name]):
                v = upd[name].to(device=device, dtype=torch.float32).view(-1)
                vecs.append(v)
        if not vecs:
            return torch.zeros(1, device=device)
        return torch.cat(vecs, dim=0)

    def fltrust_update(self, target_model, updates: dict, root_loader=None):
        """
        FLTrust aggregation:
          1) Compute server update g0 on root data (R steps, lr=server_lr)
          2) For each client update gi: TS_i = ReLU(cos(gi, g0))
          3) Normalize gi to ||g0||; aggregate weighted normalized updates.
          4) Optionally average according to fltrust_weights_norm.
        Returns: (is_updated, names, trust_scores)
        """
        if root_loader is None:
            root_loader = self.build_fltrust_root_loader()

        steps = int(self.params.get('fltrust_steps', 1))
        lr = float(self.params.get('fltrust_server_lr', 0.01))
        eta = float(self.params.get('eta', 0.1))
        no_models = int(self.params.get('no_models', 1))
        scale_like_fedavg = bool(self.params.get('fltrust_scale_like_fedavg', True))
        avg_by_clients = bool(self.params.get('fltrust_avg_by_clients', True))
        device = next(target_model.parameters()).device

        # 1) server update on root
        g0 = self._compute_server_update(target_model, root_loader, steps=steps, lr=lr)
        g0_norm = self._norm_of_update(g0)
        if not torch.isfinite(g0_norm) or g0_norm.item() < 1e-12:
            logger.info("[FLTrust] skip (invalid g0 norm)")
            return False, [], []

        # 2) collect client names and updates
        names, grads = [], []
        for name, (alpha, gi) in updates.items():
            names.append(name)
            grads.append(gi)

        # 3) compute trust scores and normalized updates
        eps = 1e-12
        g0_vec = self._vectorize_update(g0)
        g0_vn = torch.nn.functional.normalize(g0_vec, dim=0)
        trust_scores = []
        normed_updates = []

        for gi in grads:
            gi_vec = self._vectorize_update(gi)
            if gi_vec.numel() != g0_vec.numel():
                # Pad or skip mismatch - here we compute cos on intersection: fallback to full vector cosine using dot on overlapping subset is complex; use simple fallback
                min_len = min(gi_vec.numel(), g0_vec.numel())
                cos = torch.dot(gi_vec[:min_len], g0_vec[:min_len]) / (torch.norm(gi_vec[:min_len]) * torch.norm(g0_vec[:min_len]) + eps)
            else:
                gi_vn = torch.nn.functional.normalize(gi_vec, dim=0)
                cos = torch.clamp(torch.dot(gi_vn, g0_vn), -1.0, 1.0)
            ts = max(0.0, float(cos.item()))
            trust_scores.append(ts)

            # scale gi to ||g0||
            gi_norm = max(float(torch.norm(gi_vec).item()), eps)
            do_scale = bool(self.params.get('fltrust_scale_to_g0', True))
            scale = float(g0_norm.item()) / gi_norm if do_scale else 1.0
            # scaled per-layer dict
            scaled = {}
            for k, v in gi.items():
                if torch.is_floating_point(v):
                    scaled[k] = (v.to(dtype=torch.float32, device=device) * scale)
            normed_updates.append(scaled)

        ts_tensor = torch.tensor(trust_scores, dtype=torch.float32, device=device)

        gamma = float(self.params.get('fltrust_ts_gamma', 1.0))
        s_min = float(self.params.get('fltrust_trust_floor', 0.0))
        s_max = float(self.params.get('fltrust_trust_cap', 1.0))
        w = torch.clamp(ts_tensor.pow(gamma), s_min, s_max)
        if torch.sum(w).item() == 0.0:
            logger.info("[FLTrust] all trust scores zero; skip update")
            return False, names, trust_scores

        # 4) aggregate: sum_i s_i * ghat_i (NO weights sum normalization)
        norm_mode = self.params.get('fltrust_weights_norm', 'none')
        agg = {}
        for k in normed_updates[0].keys():
            acc = torch.zeros_like(normed_updates[0][k], dtype=torch.float32, device=device)
            for i, upd in enumerate(normed_updates):
                if k in upd:
                    acc.add_(upd[k] * w[i])
            if norm_mode == 'sumts':
                acc.mul_(1.0 / (float(w.sum().item()) + 1e-12))
            elif norm_mode == 'clients':
                acc.mul_(1.0 / float(len(normed_updates)))
            agg[k] = acc

        # optional clip
        agg_norm = float(self._norm_of_update(agg).item())
        max_norm = float(self.params.get('fltrust_max_update_norm', 1e9))
        if agg_norm > max_norm > 0:
            scale = max_norm / (agg_norm + 1e-12)
            for k in agg.keys():
                agg[k].mul_(scale)
            agg_norm = float(self._norm_of_update(agg).item())

        # effective eta
        eta_eff = eta / float(no_models) if scale_like_fedavg and no_models > 0 else eta
        pre_sum = sum(p.data.abs().sum().item() for _, p in target_model.named_parameters())
        ok = self._apply_aggregated_update(target_model, agg, eta=eta_eff)
        post_sum = sum(p.data.abs().sum().item() for _, p in target_model.named_parameters())
        logger.info(f"[FLTrust] delta_W={post_sum - pre_sum:.4g} (ok={ok}, eta_eff={eta_eff})")
        # logging
        nz_pct = 100.0 * float((ts_tensor > 0).float().mean().item())
        top_idx = int(torch.argmax(ts_tensor).item())
        top_pair = (names[top_idx] if names else -1, float(ts_tensor[top_idx].item()))
        logger.info(f"[FLTrust] R={steps} lr={lr} | g0={g0_norm:.3g} | TS mean={float(ts_tensor.mean().item()):.3f}, nz={nz_pct:.0f}%"
                    f", top={top_pair[0]}:{top_pair[1]:.3f} | agg={agg_norm:.3g} | eta_eff={eta_eff}")

        # optional CSV
        try:
            utils.csv_record.add_weight_result(names, trust_scores, alpha=[u[0] for u in updates.values()])
        except Exception:
            pass

        return ok, names, trust_scores

    def _compute_server_update(self, target_model, root_loader, steps: int = 1, lr: float = 0.01):
        """Compute g0 by training a server copy of target_model on root data for 'steps' steps (SGD)."""
        import copy as _copy
        server = _copy.deepcopy(target_model)
        server.train()
        device = next(server.parameters()).device
        opt = torch.optim.SGD(server.parameters(), lr=lr, momentum=self.params.get('momentum', 0.9),
                              weight_decay=self.params.get('decay', 0.0))
        it = iter(root_loader)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(max(1, steps)):
            try:
                data, target = next(it)
            except StopIteration:
                it = iter(root_loader); data, target = next(it)
            data = data.to(device); target = target.to(device)
            opt.zero_grad()
            out = server(data)
            loss = loss_fn(out, target)
            loss.backward()
            opt.step()

        # g0 = server - target
        g0 = {}
        w0 = {n: p.detach().to(device) for n, p in target_model.state_dict().items()}
        for name, p in server.state_dict().items():
            if torch.is_tensor(p) and torch.is_floating_point(p) and name in w0:
                g0[name] = (p.detach() - w0[name]).to(dtype=torch.float32)
        return g0

    # =========================
    # FedAvg-backend defenses
    # =========================
    def _apply_fedavg_backend(self, target_model, updates_list, weights=None):
        """Apply a list of client updates using FedAvg-style backend update rule."""
        if not updates_list:
            return False

        device = next(target_model.parameters()).device
        if weights is None:
            weights = [1.0 for _ in updates_list]
        if len(weights) != len(updates_list):
            logger.info('[defense] invalid weights length, fallback to uniform weights')
            weights = [1.0 for _ in updates_list]

        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        denom = float(torch.sum(weights_t).item())
        if denom <= 0:
            logger.info('[defense] sum(weights)<=0, skip update')
            return False

        agg = dict()
        for upd, w in zip(updates_list, weights_t):
            if float(w.item()) <= 0:
                continue
            for name, val in upd.items():
                if not torch.is_tensor(val) or (not torch.is_floating_point(val)):
                    continue
                vv = val.to(device=device, dtype=torch.float32)
                if name not in agg:
                    agg[name] = torch.zeros_like(vv, dtype=torch.float32, device=device)
                agg[name].add_(vv * w)

        eta = float(self.params.get('eta', 1.0))
        for name, data in target_model.state_dict().items():
            if not torch.is_floating_point(data):
                continue
            if name not in agg:
                continue
            update_per_layer = agg[name] * (eta / denom)
            if self.params.get('diff_privacy', False):
                noise = self.dp_noise(data, self.params['sigma'])
                update_per_layer = update_per_layer.to(dtype=noise.dtype, device=noise.device) + noise
            data.add_(update_per_layer.to(dtype=data.dtype, device=data.device))
        return True

    def fldetector_update(self, target_model, updates: dict):
        """
        FLDetector-style robust filtering:
          1) vectorize client updates
          2) compute robust center by coordinate-wise median
          3) drop outliers by median+MAD threshold
          4) aggregate kept updates with FedAvg backend
        Returns: (is_updated, names, scores)
        """
        if not updates:
            return False, [], []

        names = list(updates.keys())
        alphas = [updates[n][0] for n in names]
        update_list = [updates[n][1] for n in names]

        vecs = [self._vectorize_update(u) for u in update_list]
        mat = torch.stack(vecs, dim=0)
        center = torch.median(mat, dim=0).values
        dists = torch.norm(mat - center, dim=1)

        med = torch.median(dists)
        mad = torch.median(torch.abs(dists - med)) + 1e-12
        k = float(self.params.get('fldetector_mad_k', config.FLDETECTOR_DEFAULTS['fldetector_mad_k']))
        threshold = med + k * mad

        keep_mask = dists <= threshold
        if int(torch.sum(keep_mask).item()) == 0:
            keep_mask[torch.argmin(dists)] = True

        kept_updates = [u for u, m in zip(update_list, keep_mask.tolist()) if m]
        kept_names = [n for n, m in zip(names, keep_mask.tolist()) if m]
        scores = (1.0 / (1.0 + dists)).detach().cpu().tolist()
        logger.info(f"[FLDetector] kept {len(kept_names)}/{len(names)} clients: {kept_names}")

        ok = self._apply_fedavg_backend(target_model, kept_updates)
        try:
            utils.csv_record.add_weight_result(names, scores, alpha=alphas)
        except Exception:
            pass
        return ok, names, scores

    def leadfl_update(self, target_model, updates: dict):
        """
        LEADFL-style trust weighting:
          1) direction score by cosine to robust center
          2) scale score by closeness to median update norm
          3) keep top-k by trust, then weighted FedAvg backend
        Returns: (is_updated, names, trust_scores)
        """
        if not updates:
            return False, [], []

        names = list(updates.keys())
        alphas = [updates[n][0] for n in names]
        update_list = [updates[n][1] for n in names]

        vecs = [self._vectorize_update(u) for u in update_list]
        mat = torch.stack(vecs, dim=0)
        center = torch.median(mat, dim=0).values
        center = center / (torch.norm(center) + 1e-12)

        norms = torch.norm(mat, dim=1) + 1e-12
        norm_med = torch.median(norms)
        cos = F.cosine_similarity(mat, center.unsqueeze(0).expand_as(mat), dim=1)
        dir_score = torch.clamp(cos, min=0.0)
        scale_score = torch.exp(-torch.abs(torch.log(norms / (norm_med + 1e-12))))
        gamma = float(self.params.get('leadfl_gamma', config.LEADFL_DEFAULTS['leadfl_gamma']))
        trust = torch.clamp((dir_score * scale_score).pow(gamma), min=0.0)

        keep_ratio = float(self.params.get('leadfl_keep_ratio', config.LEADFL_DEFAULTS['leadfl_keep_ratio']))
        keep_ratio = min(1.0, max(0.1, keep_ratio))
        keep_k = max(1, int(math.ceil(len(names) * keep_ratio)))
        top_vals, top_idx = torch.topk(trust, k=keep_k)

        keep_set = set(top_idx.detach().cpu().tolist())
        kept_updates = []
        kept_weights = []
        kept_names = []
        w_floor = float(self.params.get('leadfl_weight_floor', config.LEADFL_DEFAULTS['leadfl_weight_floor']))
        for i, (n, upd) in enumerate(zip(names, update_list)):
            if i in keep_set:
                kept_updates.append(upd)
                kept_names.append(n)
                kept_weights.append(max(float(trust[i].item()), w_floor))

        if len(kept_updates) == 0:
            best_i = int(torch.argmax(trust).item())
            kept_updates = [update_list[best_i]]
            kept_names = [names[best_i]]
            kept_weights = [1.0]

        logger.info(f"[LEADFL] kept {len(kept_names)}/{len(names)} clients: {kept_names}")
        ok = self._apply_fedavg_backend(target_model, kept_updates, weights=kept_weights)
        trust_scores = trust.detach().cpu().tolist()
        try:
            utils.csv_record.add_weight_result(names, trust_scores, alpha=alphas)
        except Exception:
            pass
        return ok, names, trust_scores

    def flip_update(self, target_model, updates: dict):
        """
        FLIP defense:
          1) detect sign-flipped updates by cosine to mean direction
          2) for suspicious updates, flip sign back
          3) clip per-client norm to a robust cap and aggregate with FedAvg backend
        Returns: (is_updated, names, scores)
        """
        if not updates:
            return False, [], []

        names = list(updates.keys())
        alphas = [updates[n][0] for n in names]
        update_list = [updates[n][1] for n in names]

        vecs = [self._vectorize_update(u) for u in update_list]
        mat = torch.stack(vecs, dim=0)
        ref = torch.mean(mat, dim=0)
        cos = F.cosine_similarity(mat, ref.unsqueeze(0).expand_as(mat), dim=1)

        cos_th = float(self.params.get('flip_cos_threshold', config.FLIP_DEFAULTS['flip_cos_threshold']))
        norm_cap_ratio = float(self.params.get('flip_max_norm_ratio', config.FLIP_DEFAULTS['flip_max_norm_ratio']))
        norms = torch.norm(mat, dim=1)
        norm_cap = float(torch.median(norms).item()) * max(norm_cap_ratio, 1.0)

        device = next(target_model.parameters()).device
        corrected = []
        corrected_cnt = 0
        for i, upd in enumerate(update_list):
            sign = -1.0 if float(cos[i].item()) < cos_th else 1.0
            if sign < 0:
                corrected_cnt += 1

            cur = dict()
            sq_sum = 0.0
            for k, v in upd.items():
                if not torch.is_tensor(v) or (not torch.is_floating_point(v)):
                    continue
                vv = v.to(device=device, dtype=torch.float32) * sign
                cur[k] = vv
                sq_sum += float(torch.sum(vv * vv).item())

            cur_norm = math.sqrt(sq_sum + 1e-12)
            if norm_cap > 0 and cur_norm > norm_cap:
                scale = norm_cap / (cur_norm + 1e-12)
                for k in cur.keys():
                    cur[k].mul_(scale)

            corrected.append(cur)

        logger.info(f"[FLIP] corrected {corrected_cnt}/{len(names)} potentially sign-flipped clients")
        ok = self._apply_fedavg_backend(target_model, corrected)
        scores = torch.clamp((cos + 1.0) * 0.5, min=0.0, max=1.0).detach().cpu().tolist()
        try:
            utils.csv_record.add_weight_result(names, scores, alpha=alphas)
        except Exception:
            pass
        return ok, names, scores


    # =========================
    # Experiment metrics helpers
    # =========================
    def reset_round_stats(self):
        """Reset counters that are accumulated within one server round (epoch)."""
        self.round_stats = {
            # data efficiency
            'seen_samples': 0,
            'dropped_samples': 0,
            'poison_injected': 0,
            'poison_dropped': 0,
            'clean_dropped': 0,
            # online prefilter overhead
            'prefilter_scored': 0,
            'prefilter_dropped': 0,
            'prefilter_time_s': 0.0,
        }

    def _rs_inc(self, key: str, val):
        """Increment a round stat (safe for missing keys)."""
        if val is None:
            return
        if key not in self.round_stats:
            self.round_stats[key] = 0
        self.round_stats[key] += val

    def snapshot_round_stats(self) -> dict:
        """Return a copy of current round stats with a few derived rates."""
        s = dict(self.round_stats) if hasattr(self, 'round_stats') else {}
        seen = float(s.get('seen_samples', 0) or 0)
        dropped = float(s.get('dropped_samples', 0) or 0)
        scored = float(s.get('prefilter_scored', 0) or 0)
        pd = float(s.get('prefilter_dropped', 0) or 0)
        pinj = float(s.get('poison_injected', 0) or 0)
        pdrop = float(s.get('poison_dropped', 0) or 0)
        s['drop_rate_total'] = (dropped / seen) if seen > 0 else 0.0
        s['prefilter_drop_rate'] = (pd / scored) if scored > 0 else 0.0
        s['poison_drop_rate'] = (pdrop / pinj) if pinj > 0 else 0.0
        return s

    def log_round_metrics(self, epoch: int, **metrics):
        """Append one JSONL record for this round."""
        payload = {
            'epoch': int(epoch),
            'time': time.time(),
            'exp_name': self.params.get('name', self.name),
            'dataset': self.params.get('dataset', self.params.get('type', '')),
            'dirichlet_alpha': self.params.get('dirichlet_alpha', None),
        }
        payload.update(metrics)
        payload.update(self.snapshot_round_stats())
        try:
            with open(self._metrics_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.info(f'[METRICS] failed to write metrics jsonl: {e}')

class FoolsGold(object):
    def __init__(self, use_memory=False):
        self.memory = None
        self.memory_dict=dict()
        self.wv_history = []
        self.use_memory = use_memory

    def aggregate_gradients(self, client_grads,names):
        cur_time = time.time()
        num_clients = len(client_grads)
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()

        # if self.memory is None:
        #     self.memory = np.zeros((num_clients, grad_len))
        self.memory = np.zeros((num_clients, grad_len))
        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
            if names[i] in self.memory_dict.keys():
                self.memory_dict[names[i]]+=grads[i]
            else:
                self.memory_dict[names[i]]=copy.deepcopy(grads[i])
            self.memory[i]=self.memory_dict[names[i]]
        # self.memory += grads

        if self.use_memory:
            wv, alpha = self.foolsgold(self.memory)  # Use FG
        else:
            wv, alpha = self.foolsgold(grads)  # Use FG
        logger.info(f'[foolsgold agg] wv: {wv}')
        self.wv_history.append(wv)

        agg_grads = []
        # Iterate through each layer
        for i in range(len(client_grads[0])):
            assert len(wv) == len(client_grads), 'len of wv {} is not consistent with len of client_grads {}'.format(len(wv), len(client_grads))
            temp = wv[0] * client_grads[0][i].cpu().clone()
            # Aggregate gradients for a layer
            for c, client_grad in enumerate(client_grads):
                if c == 0:
                    continue
                temp += wv[c] * client_grad[i].cpu()
            temp = temp / len(client_grads)
            agg_grads.append(temp)
        print('model aggregation took {}s'.format(time.time() - cur_time))
        return agg_grads, wv, alpha

    def foolsgold(self,grads):
        """
        :param grads:
        :return: compute similatiry and return weightings
        """
        n_clients = grads.shape[0]
        cs = smp.cosine_similarity(grads) - np.eye(n_clients)

        maxcs = np.max(cs, axis=1)
        # pardoning
        for i in range(n_clients):
            for j in range(n_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        wv = 1 - (np.max(cs, axis=1))

        wv[wv > 1] = 1
        wv[wv < 0] = 0

        alpha = np.max(cs, axis=1)

        # Rescale so that max value is wv
        wv = wv / np.max(wv)
        wv[(wv == 1)] = .99

        # Logit function
        wv = (np.log(wv / (1 - wv)) + 0.5)
        wv[(np.isinf(wv) + wv > 1)] = 1
        wv[(wv < 0)] = 0

        # wv is the weight
        return wv,alpha
