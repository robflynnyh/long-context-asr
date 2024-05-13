''' 
Single utterance self-training wrapper
Given a model it wraps the forward pass with logic that performs n iterations, updating the weights based on
pseudo-label loss, new weights are discarded after iterations are done.
'''
from lcasr.optim.madgrad import MADGRAD
from lcasr.utils.augmentation import SpecAugment
from lcasr.decoding.greedy import GreedyCTCDecoder
from functools import partial
import torch

def su_selftrain_wrapper(
        model, 
        prediction_key="final_posteriors", 
        n_iterations=10, 
        lr=9e-5,
        spec_augment_config={
            'n_time_masks': 0,
            'n_freq_masks': 6,
            'freq_mask_param': 34,
        }
    ):
    model.forward_loop = model.forward
    ctc_loss_fn = torch.nn.CTCLoss(blank=model.decoder.num_classes-1, reduction='sum')
    augmentation = SpecAugment(**spec_augment_config) 
    decoder = GreedyCTCDecoder(tokenizer = None, blank_id = model.decoder.num_classes-1)
    decoder.forward = partial(decoder.forward, decode=False)
    original_model_params = list(model.parameters())
    original_model_params = [p.clone().detach().cpu() for p in original_model_params]
    
    @torch.set_grad_enabled(True)
    def forward_wrapper(*args, **kwargs):
        optimizer = MADGRAD(model.parameters(), lr=lr)
        original_audio_signal = args[0].clone()
        model.eval()
        for i in range(n_iterations):
            if i < n_iterations - 1:
                model.zero_grad()
                audio_signal = torch.cat([
                    augmentation(original_audio_signal),
                    original_audio_signal
                ], dim=0)
                args_i = (audio_signal, *args[1:])
                outputs = model.forward_loop(*args_i, **kwargs)
                predictions = outputs[prediction_key]
                pseudo_labels = torch.LongTensor(decoder(predictions)[-1])[None].to(predictions.device)
              
                pseudo_loss = ctc_loss_fn(
                    predictions[0][None].transpose(0, 1), 
                    pseudo_labels, 
                    input_lengths = torch.LongTensor([predictions.size(1)]).to(predictions.device),
                    target_lengths = torch.LongTensor([pseudo_labels.size(1)]).to(predictions.device)
                ) / predictions.size(1)
          
                pseudo_loss.backward()
                optimizer.step()
            else:
                model.zero_grad()
                outputs = model.forward_loop(*args, **kwargs)

        for p, p_orig in zip(model.parameters(), original_model_params):
            p.data = p_orig.data.to(p.device)

        return outputs
    
    model.forward = forward_wrapper
    return model




            
            
