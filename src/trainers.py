# generic
import os
import pickle
import torch
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
from collections import defaultdict
import re
import glob

# models
from models_bit_diff import BitDiffPredictorTCN
from bit_diffusion import GaussianBitDiffusion
from ema import *

# eval
from evaluation import *



class TrainerTCN:
    def __init__(self, args, causal=False):
        super(TrainerTCN, self).__init__()

        # PARAMS
        self.num_stages = args.num_stages
        self.num_classes = args.num_classes
        self.m_name = args.model
        self.input_dim = args.input_dim

        if self.m_name == 'bit-diff-pred-tcn':
            self.model = BitDiffPredictorTCN(args, causal=causal)
            self.model_dim = args.model_dim
    
            self.diffusion = GaussianBitDiffusion(
                self.model, 
                condition_x0=args.conditioned_x0,
                num_classes=self.num_classes,
                timesteps=args.num_diff_timesteps,
                ddim_timesteps=args.num_infr_diff_timesteps,
                objective=args.diff_obj,
                loss_type=args.diff_loss_type,
            )

    # ------------------------------------------------------------------- TRAINING ---------------------------------------------------------------
    def train(self,
              args,
              save_dir,
              batch_gen,
              val_batch_gens,
              device,
              num_workers,
              writer,
              results_dir,
              actions_dict):

        # MODEL
        self.diffusion = self.diffusion.to(device)
        self.diffusion.train()
        self.model.train()
        self.ema_diffusion = EMA(model=self.diffusion,
                                beta=0.995,
                                update_every=10)
        self.ema_diffusion.eval()


        # INIT OPTIMIZER & SCHEDULER
        optimizer = optim.Adam(params=self.diffusion.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5)

        start_epoch = 0
        if args.load_best:
            print("Searching for previous checkpoints...")
            # Find all directories matching "epoch-*" in save_dir
            epoch_dirs = glob.glob(os.path.join(save_dir, "epoch-*"))
            valid_epochs = []
            
            for edir in epoch_dirs:
                if os.path.exists(os.path.join(edir, "checkpoint.pth")):
                    # Extract the epoch number from the folder name
                    match = re.search(r'epoch-(\d+)', edir)
                    if match:
                        valid_epochs.append((int(match.group(1)), edir))
            
            if valid_epochs:
                # Sort by epoch number and grab the highest one
                valid_epochs.sort(key=lambda x: x[0])
                best_epoch, best_dir = valid_epochs[-1]
                print(f"Found checkpoint! Resuming from epoch {best_epoch} at {best_dir}...")
                
                # Load the bundled state dicts
                checkpoint = torch.load(os.path.join(best_dir, "checkpoint.pth"), map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
                self.ema_diffusion.load_state_dict(checkpoint['ema_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Set start epoch to the next epoch so we don't repeat work
                start_epoch = best_epoch + 1 
            else:
                print("No valid checkpoints found. Starting from scratch.")

        # TRAIN
        print('Start Training...')
        for epoch in range(start_epoch, args.num_epochs + 1):
            epoch_loss = 0
            epoch_ce_loss = 0

            correct, total = 0, 0
            correct_past, total_past = 0, 0
            correct_future, total_future = 0, 0

            dataloader = DataLoader(
                batch_gen,
                batch_size=args.bz,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=batch_gen.custom_collate
            )

            for itr, sample_batched in enumerate(dataloader):
                batch_total_loss, batch_ce_loss, \
                batch_correct, batch_total, \
                batch_correct_past, batch_total_past, \
                batch_correct_future, batch_total_future = self.train_single_batch(sample_batched, optimizer, device)

                # losses
                epoch_loss += batch_total_loss
                epoch_ce_loss += batch_ce_loss

                # acc
                total += batch_total
                correct += batch_correct

                total_past += batch_total_past
                correct_past += batch_correct_past

                total_future += batch_total_future
                correct_future += batch_correct_future

                # log
                if itr % 5 == 0:
                    print("[epoch %f]: epoch loss = %f, ce loss = %f, past acc = %f, fut acc = %f " % (epoch + itr / len(dataloader),
                                                                                                    epoch_loss / (itr + 1),
                                                                                                    epoch_ce_loss / (itr + 1),
                                                                                                    float(correct_past) / total_past,
                                                                                                    float(correct_future) / total_future))
                self.ema_diffusion.update()    

           
            # LR step
            lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate : {lr}')
            scheduler.step(epoch_loss)

            # Logging
            print("[epoch %d]: epoch loss = %f,  acc = %f" % (epoch, epoch_loss / len(dataloader), float(correct) / total))

            # acc
            writer.add_scalar("training_accuracies/MoF_past", float(correct_past) / total_past, global_step=epoch)
            writer.add_scalar("training_accuracies/MoF_future", float(correct_future) / total_future, global_step=epoch)
            writer.add_scalar("training_accuracies/MoF_all", float(correct) / total, global_step=epoch)

            # loss
            writer.add_scalar("training_losses/total_loss",  float(epoch_loss) / len(dataloader), global_step=epoch)
            writer.add_scalar("training_losses/ce_loss", float(epoch_ce_loss) / len(dataloader), global_step=epoch)

            # SAVE RES
            if (epoch) % 5 == 0: 
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
                torch.save(self.ema_diffusion.state_dict(), save_dir + "/ema_diff_epoch-" + str(epoch) + ".model")

                epoch_dir = os.path.join(save_dir, f"epoch-{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'diffusion_state_dict': self.diffusion.state_dict(),
                    'ema_state_dict': self.ema_diffusion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                
                # Save the master checkpoint
                torch.save(checkpoint, os.path.join(epoch_dir, "checkpoint.pth"))


    def train_single_batch(self, sample_batched, optimizer, device):
        # INPUT
        # DET
        features_tensor = sample_batched[0]
        classes_tensor = sample_batched[1]

        # PROB
        classes_one_hot_tensor = sample_batched[3]

        mask_tensor = sample_batched[4]
        mask_past_tensor = sample_batched[5]
        mask_future_tensor = sample_batched[6]

        # DEVICE
        features_tensor = features_tensor.to(device)
        classes_tensor = classes_tensor.to(device)

        mask_tensor = mask_tensor.to(device)
        mask_past_tensor = mask_past_tensor.to(device)
        mask_future_tensor = mask_future_tensor.to(device)
        
        classes_one_hot_tensor = classes_one_hot_tensor.to(device)
 
        # MASKS
        masks = [mask_tensor]  # 1 stage only
 
        ''' PREDICTION '''
        loss, tcn_predictions = self.diffusion({'mask_past': rearrange(mask_past_tensor, 'b c t -> b t c'),
                                                'x_0': rearrange(classes_one_hot_tensor, 'b c t -> b t c'),
                                                'obs': rearrange(features_tensor, 'b c t -> b t c'),
                                                'masks_stages': [rearrange(mt, 'b c t -> b t c') for mt in masks]})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ''' RESULTS '''
        # LOSSES
        batch_loss = loss.item()
        ce_loss  = torch.tensor(0.)            
        ce_loss = ce_loss.item()

        # ACCURASIES
        _, predicted = torch.max(tcn_predictions[-1].data, 1)

        correct = ((predicted == classes_tensor).float() * mask_tensor.squeeze(1)).sum().item()
        correct_past = ((predicted == classes_tensor).float() * mask_past_tensor.squeeze(1)).sum().item()
        correct_future = ((predicted == classes_tensor).float() * mask_future_tensor.squeeze(1)).sum().item()

        total_future = torch.sum(mask_future_tensor).item()
        total_past = torch.sum(mask_past_tensor).item()
        total = torch.sum(mask_tensor).item()

        return batch_loss, ce_loss, \
               correct, total, correct_past, total_past, \
               correct_future, total_future


    # ------------------------------------------------------------------ VALIDATION ---------------------------------------------------------------
    def validate(self,
                 args,
                 epoch,
                 obs_perc,
                 batch_gen,
                 device,
                 num_workers,
                 model_dir,
                 actions_dict,
                 sample_rate,
                 eval_mode):

        with torch.no_grad():
            print(f'Obs perc : {obs_perc}\n')

            # MODEL
            if not eval_mode:
                self.model.to(device)
                self.model.eval()
                self.model.load_state_dict(torch.load(model_dir + '/epoch-' + str(epoch) + ".model"))

                # diff
                self.diffusion.to(device)
                self.diffusion.eval()
                self.diffusion.model.load_state_dict(torch.load(model_dir + '/epoch-' + str(epoch) + ".model"), strict=True)

                # ema
                self.ema_diffusion = EMA(model=self.diffusion,
                                        beta=0.995,
                                        update_every=10)
                self.ema_diffusion.load_state_dict(torch.load(model_dir + "/ema_diff_epoch-" + str(epoch) + ".model"), strict=False)
                self.ema_diffusion.eval()

            else:
                self.model.eval()
                self.diffusion.eval()
                self.ema_diffusion.eval()


            # PERCS
            eval_percentages = [.1, .2, .3, .5]

            # LOGGERS
            # Metric loggers
            n_T_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            n_F_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            max_n_T_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            max_n_F_classes_all_files = np.zeros((len(eval_percentages), len(actions_dict)))
            
            num_frames_all_files = np.zeros(len(eval_percentages))
            errors_frames_all_files = np.zeros(len(eval_percentages))
            
            loss, ce_loss = 0, 0          
            
            # EVAL
            dataloader = DataLoader(
                batch_gen, 
                batch_size=1, 
                shuffle=False, 
                num_workers=args.num_workers, 
                collate_fn=batch_gen.custom_collate
            )

            # SAVE (DIFF PREDS)
            if args.ds not in ['bf']:
                ds = args.ds
            else:
                ds = f'{args.ds}_{args.split}'

            result_dict = defaultdict(list)
            result_file = f'./diff_results/{ds}/{args.model}'\
                            f'_epoch_{epoch}'\
                            f'_lt_{args.layer_type}'\
                            f'_ns_{args.num_stages}'\
                            f'_nl_{args.num_layers}'\
                            f'_ds_{args.num_diff_timesteps}'\
                            f'_ids_{args.num_infr_diff_timesteps}'\
                            f'_num_samples_{args.num_samples}' \
                            f'_dlt_{args.diff_loss_type}' \
                            f'_cond_x0_{args.conditioned_x0}'
            
            if not os.path.exists(result_file):
                os.makedirs(result_file)
            result_file += f'/obs_{obs_perc}.pkl'


            # ITERATE
            for itr, sample_batched in enumerate(dataloader):
                # DATA
                features = sample_batched[0] 

                classes_tensor = sample_batched[1]  
                classes_all_tensor = sample_batched[2]
                classes_one_hot_tensor = sample_batched[3] 

                mask_past_tensor = sample_batched[5]

                if args.qualitative:
                    keep_lowdim = (classes_tensor != args.ignore_action).squeeze()
                    keep_highdim = (classes_all_tensor != args.ignore_action).squeeze()

                    features = features[..., keep_lowdim]
                    classes_tensor = classes_tensor[..., keep_lowdim]
                    classes_all_tensor = classes_all_tensor[..., keep_highdim]
                    classes_one_hot_tensor = classes_one_hot_tensor[..., keep_lowdim]
                    mask_past_tensor = mask_past_tensor[..., keep_lowdim]

                    print(features.shape, classes_tensor.shape,
                        classes_all_tensor.shape, classes_one_hot_tensor.shape,
                        mask_past_tensor.shape)
                    
                    print(classes_tensor)


                # META INFO
                init_vid_len = sample_batched[-2]
                meta_dict = sample_batched[-1]
                file_names = meta_dict['file_names']    


                # DEVICE
                features = features.to(device)
                classes_tensor = classes_tensor.to(device)
                classes_one_hot_tensor = classes_one_hot_tensor.to(device)
                mask_past_tensor = mask_past_tensor.to(device)


                # MASK
                masks = [torch.ones(1, 1, features.size(-1), device=device)]


                ''' PREDICTIONS '''
                # PROB
                tcn_predictions = self.ema_diffusion.ema_model.predict(
                        x_0 = rearrange(classes_one_hot_tensor, 'b c t -> b t c'),  # used only for shape
                        obs = rearrange(features, 'b c t -> b t c'),
                        mask_past = rearrange(mask_past_tensor, 'b c t -> b t c'),
                        masks_stages = [rearrange(mask_tensor, 'b c t -> b t c') for mask_tensor in masks],
                        n_samples=args.num_samples,
                        n_diffusion_steps=args.num_infr_diff_timesteps
                        )
                tcn_predictions = tcn_predictions.contiguous()  # N x B x C x T (N - number of samples)
                loss = 0.
                ce_loss = 0.


                ''' ACCURACIES '''
                gt_content = classes_all_tensor[0].numpy()
                
                if args.qualitative:
                    init_vid_len = len(classes_all_tensor[0])

                assert len(gt_content) == init_vid_len

                # ACCUM PREDICTIONS
                # iterate through samples
                tcn_fin_predictions = []
                for s in range(tcn_predictions.shape[0]):  # N
                    tcn_fin_prediction = []
                    _, predicted = torch.max(tcn_predictions[s].data, 1)  # B x C x T --> B x T
                    predicted = predicted.squeeze()  # B x T --> T
                    for i in range(len(predicted)):
                        tcn_fin_prediction = np.concatenate((tcn_fin_prediction, [predicted[i].item()] * sample_rate))

                    # save / accumulate
                    result_dict[file_names[0]].append(tcn_fin_prediction)  # T'
                    tcn_fin_predictions.append(tcn_fin_prediction)  # T'

                result_dict[f'gt_{file_names[0]}'] = classes_all_tensor[0].numpy()
                tcn_fin_predictions = np.stack(tcn_fin_predictions, axis=0)  # N x T'

                if args.qualitative:
                    torch.save(tcn_fin_predictions, "test/sliced_sample.pt")
                else:
                    torch.save(tcn_fin_predictions, "test/full_sample.pt")
                # COMPUTE EVAL METRICS
                past_len = int(obs_perc * init_vid_len)  # observation length
                for i in range(len(eval_percentages)):
                    eval_perc = eval_percentages[i]
                    eval_len = int((eval_perc + obs_perc) * init_vid_len)

                    # Top-1 MoC
                    max_n_T_classes = np.zeros(self.num_classes)
                    max_n_F_classes = np.zeros(self.num_classes) 
                    max_moc = -1.

                    # go through N generated samples
                    for s in range(tcn_fin_predictions.shape[0]):  # N
                        _, _, classes_n_T, classes_n_F, _ = eval_file(
                            gt_content,
                            tcn_fin_predictions[s][:eval_len],
                            past_len,
                            actions_dict
                        )
                        n_T_classes_all_files[i] += classes_n_T
                        n_F_classes_all_files[i] += classes_n_F

                        # choose the max one
                        all_pred = classes_n_T + classes_n_F
                        moc = np.mean(classes_n_T[all_pred != 0] / (classes_n_T[all_pred != 0] + classes_n_F[all_pred != 0]))
                        if moc > max_moc:
                            max_moc = moc   
                            max_n_T_classes = classes_n_T        
                            max_n_F_classes = classes_n_F

                    max_n_T_classes_all_files[i] += max_n_T_classes
                    max_n_F_classes_all_files[i] += max_n_F_classes


            ''' SAVE SAMPLED RESULTS '''
            result_file_ptr = open(result_file, 'wb')
            pickle.dump(result_dict, result_file_ptr)
            result_file_ptr.close()


            ''' ACCUMULATE RESULTS '''
            loss = loss / len(dataloader)
            ce_loss = ce_loss / len(dataloader)
            results = f"Loss : {round(loss, 4)}, CE Loss : {round(ce_loss, 4)}\n\n"

            returned_metrics = []
            for j in range(len(eval_percentages)):
                acc, n = 0, 0  # Mean MoC
                max_acc, max_n = 0, 0  # Top-1 MoC

                # MoC (Mean over Classes metric)
                for i in range(len(actions_dict)):
                    if (n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i]) != 0:
                        acc += (float(n_T_classes_all_files[j, i]) / (n_T_classes_all_files[j, i] + n_F_classes_all_files[j, i]))
                        n += 1

                    if (max_n_T_classes_all_files[j, i] + max_n_F_classes_all_files[j, i]) != 0:
                        max_acc += (float(max_n_T_classes_all_files[j, i]) / (max_n_T_classes_all_files[j, i] + max_n_F_classes_all_files[j, i]))
                        max_n += 1  

                #  NORMALIZE
                mof = 100 * (1.0 - float(errors_frames_all_files[j]) / num_frames_all_files[j])
                moc = float(acc) / n
                max_moc = float(max_acc) / max_n   

                # LOG
                results += "Pred perc " + str(int(100 * eval_percentages[j])) + \
                           " , Acc: %.4f" % (100 * (1.0 - float(errors_frames_all_files[j]) / num_frames_all_files[j])) + \
                           " , MoC:  %.4f" % (float(acc) / n) + \
                           ", Max MoC:  %.4f\n" % (float(max_acc) / max_n)
                
                
                returned_metrics.append(
                    [
                        eval_percentages[j], \
                        mof, \
                        moc, \
                        max_moc, \
                        loss, \
                        ce_loss
                    ]
                )

            if not eval_mode:
                print(results)
                print()

            # print acquired results
            file = result_file.replace('.pkl', f'_results_{obs_perc}.txt')
            file = open(file, 'w')
            file.write(results + '\n')
            file.close()

        return returned_metrics


0
