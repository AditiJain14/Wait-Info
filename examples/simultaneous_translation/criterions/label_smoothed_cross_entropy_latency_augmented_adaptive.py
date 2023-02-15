# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import random
from random import choice
import numpy as np
import torch
import logging

from examples.simultaneous_translation.utils.latency import (
    LatencyTraining
)

def read2columns(file_name, split=' '):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split(split)
            x.append(p[0])
            y.append(p[1].strip())

    return zip(x, y)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    # import ipdb; ipdb.set_trace()
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    #import ipdb; ipdb.set_trace()
    return loss, nll_loss


@register_criterion('latency_augmented_label_smoothed_cross_entropy_adaptive')
class LatencyAugmentedLabelSmoothedCrossEntropyCriterionAdaptive(FairseqCriterion):

    def __init__(self, task, 
        sentence_avg, 
        label_smoothing, 
        latency_weight_avg,
        latency_weight_avg_type,
        latency_weight_var,
        latency_weight_var_type,
        mass_preservation,
        average_method,
        ignore_prefix_size=0,
        report_accuracy=False,
        adaptive_training=False,
        dict_file="dict.txt",
        adaptive_method="exp",
        adaptive_T=1.0,
        weight_drop=0.1,):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.latency_weight_avg = latency_weight_avg
        self.latency_weight_avg_type = latency_weight_avg_type
        self.latency_weight_var = latency_weight_var
        self.latency_weight_var_type = latency_weight_var_type
        self.mass_preservation = mass_preservation
        self.average_method = average_method
        self.latency_train = LatencyTraining(
            self.latency_weight_avg,
            self.latency_weight_var,
            self.latency_weight_avg_type,
            self.latency_weight_var_type,
            self.mass_preservation,
            self.average_method,
        )
        self.cuda = torch.cuda.is_available() and not self.task.args.cpu
        self.adaptive_training = False
        if adaptive_training:
            self.adaptive_training = True
            self.weight_drop = weight_drop
            freq = []
            for x, y in read2columns(dict_file, split=' '):
                freq.append(int(y))
            freq = torch.tensor(freq)
            mid = freq[int(len(freq) / 2)]
            if adaptive_method == 'exp':
                # exponential
                self.weight = [torch.exp(-1 * adaptive_T * item / mid) for item in freq]
                self.weight = torch.tensor(self.weight)
                b = self.weight.max()
                self.weight = self.weight / b * (np.e - 1) + 1
            else:
                # chi square
                self.weight = [torch.pow(item / mid, torch.tensor(2)) * torch.exp(-1 * adaptive_T * item / mid) for item in freq]
                self.weight = torch.tensor(self.weight)
                b = self.weight.max()
                self.weight = self.weight / b * (np.e - 1) + 1
            self.weight = torch.cat([torch.tensor([1., 1., 1., 1.]), self.weight], dim=0)
            if self.cuda:
                self.weight = self.weight.cuda()
            if self.task.args.fp16:
                self.weight = self.weight.half()
            
            logger.info("--> Computing Exponential-Adaptive Loss ..")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument("--latency-weight-avg", default=0., type=float, metavar='D',
                            help="Average loss weight")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="Variance loss weight")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="Statistics for Average loss type")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="Statistics for variance loss type")
        parser.add_argument("--average-method", default="weighted_average",
                            help="Average loss type")
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--adaptive-training', action='store_true',
                            help='if set, start token-level adaptive training.')
        parser.add_argument('--dict-file', default='dict.txt',
                            help='the target dictionary produced by fairseq itself.')
        parser.add_argument('--adaptive-method', default='exp', choices=['exp', 'k2'],
                            help='two methods mentioned in the paper')
        parser.add_argument('--adaptive-T', default=1., type=float,
                            help='The hyperparameter T.')
        parser.add_argument('--weight-drop', default=0.1, type=float,
                            help='A useful trick for adaptive training.')
        # fmt: on

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        flag=False
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
        
        
        loss, nll_loss, latency_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        logging_output = {
            "loss": loss.data,
        "latency_loss": latency_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        #import ipdb; ipdb.set_trace()
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # if getattr(lprobs, "batch_first", False):
            #     lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            #     target = target[:, self.ignore_prefix_size :].contiguous()
            # else:
            #     lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
            #     target = target[self.ignore_prefix_size :, :].contiguous()
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def vanilla_compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss
    
    def compute_loss(self, model, net_output, sample, reduce=True):
        # Compute cross entropy loss first
        if self.adaptive_training and self.training:
            loss, nll_loss = self.compute_adaptive_loss(model, net_output, sample, reduce=reduce)
        else:
            # original label-smoothed cross entropy loss
            loss, nll_loss = self.vanilla_compute_loss(model, net_output, sample, reduce=reduce)

        # Obtain the expected alignment
        attn_list = [item["alpha"] for item in net_output[-1]["attn_list"]]

        target_padding_mask = model.get_targets(sample, net_output).eq(self.padding_idx)

        source_padding_mask = net_output[-1].get("encoder_padding_mask", None)

        # Get latency loss
        latency_loss = self.latency_train.loss(
            attn_list, source_padding_mask, target_padding_mask)

        loss += latency_loss

        return loss, nll_loss, latency_loss
#old function
    # def vanilla_compute_loss(self, model, net_output, sample, reduce=True):
    #     lprobs = model.get_normalized_probs(net_output, log_probs=True)
    #     lprobs = lprobs.view(-1, lprobs.size(-1))
    #     target = model.get_targets(sample, net_output).view(-1, 1)
    #     loss, nll_loss = label_smoothed_nll_loss(
    #         lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
    #     )
    #     return loss, nll_loss
    def compute_adaptive_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        target = target.view(-1,1)
        non_pad_mask = target.ne(self.padding_idx)
        loss_weight = self.weight[target]
        drop_p = self.weight_drop * torch.ones_like(loss_weight)
        drop_mask = torch.bernoulli(drop_p).byte()
        loss_weight.masked_fill_(drop_mask, 1.)
        nll_loss = -(loss_weight * (lprobs.gather(dim=-1, index=target)))[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss
    
    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        latency_loss_sum = sum(log.get("latency_loss", 0) for log in logging_outputs)
        #import ipdb; ipdb.set_trace()
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        if(loss_sum!=0 or latency_loss_sum!=0 or nll_loss_sum!=0):
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )
            metrics.log_scalar(
                "latency_loss",
                latency_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
            metrics.log_scalar(
                "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )

            total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
            if total > 0:
                metrics.log_scalar("total", total)
                n_correct = utils.item(
                    sum(log.get("n_correct", 0) for log in logging_outputs)
                )
                metrics.log_scalar("n_correct", n_correct)
                metrics.log_derived(
                    "accuracy",
                    lambda meters: round(
                        meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                    )
                    if meters["total"].sum > 0
                    else float("nan"),
                )
            
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
