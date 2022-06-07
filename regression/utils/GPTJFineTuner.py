import openai, os, time, torch, sys, importlib, json, copy
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import numpy as np
sys.path.insert(1, '/home/user0/Desktop/LIFT-arXiv')
from gptj import lora_gptj
from gptj.lora_gptj import AverageMeter
from resultsCollector import regressionLoss
from collections import defaultdict

def L2error(y1, y2):
    try:
        return np.linalg.norm(y1.reshape(-1) - y2.reshape(-1))
    except AttributeError:
        try:
            return np.linalg.norm(y1 - y2.reshape(-1))
        except AttributeError:
            try:
                return np.linalg.norm(y1.reshape(-1) - y2)
            except AttributeError:
                return np.linalg.norm(y1 - y2)

def RMSE(a,b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    return np.mean((a-b)**2)**0.5


def RMSE_woo(a,b,threshold=20):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise ValueError('RMSE input error')
    std = RMSE(a,b)
    outlier_flag = (np.abs(a-b) > std*threshold)
    num_outlier = np.sum(outlier_flag)
    
    return RMSE(a[~outlier_flag],b[~outlier_flag]), num_outlier

class GPTJFineTuner(object):
    def __init__(self,config:dict,train_jsonl,valid_jsonl,cuda_idx = 0):
        self.config = config
        self.train_jsonl=train_jsonl
        self.valid_jsonl=valid_jsonl

        self.device = torch.device('cuda:%d' % cuda_idx) if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(cuda_idx)
    
    def init_model(self):
        self.ft_model = lora_gptj.LoRaQGPTJ(adapter=True, device=self.device)

    def fine_tune(self):
        self.init_model()

    def generate(self, gpt, text_lst, max_token=10, batch_size=2):
        gpt.model.eval()
        outputs = []
        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = gpt.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
            prompt = {key: value.to(gpt.device) for key, value in prompt.items()}
            outs = gpt.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=gpt.tokenizer.eos_token_id, do_sample=True, early_stopping = True)
            outs = gpt.tokenizer.batch_decode(outs, skip_special_tokens=True)
            outputs += outs
        return outputs
        
    def prompt2value(self, x, valid_mean):
        # print("Output:",x)
        c = x.strip().split('@@@')[0]
        if c == '':
            return valid_mean
        try:
            return float(c)
        except:
            return valid_mean
    
    def query(self, gpt, prompts, bs=10, valid_mean = 0):
        outputs = self.generate(gpt, prompts, batch_size=bs)
        ans = []
        for txt in outputs:
            try:
                output = self.prompt2value(txt.split('@@@')[0].split('###')[-1], valid_mean)
            except:
                output = valid_mean
            ans.append(output)
        return ans


    def eval(self,n_train,valid_prompts,valid_df,test_prompts,test_df,y_name='y',plot=False,X_grid=None,grid_prompts=None,y_grid=None,file_name=None, train_df = None, log = False):
        """
            number of valid samples
            L2 error on the valid samples
        """
        valid_mean = train_df.y.mean()
        y_valid_outputs_,len_valid_valid_y_, rmse_, rmse_woo_, y_test_outputs_ = [], [], [], [], []
        best_idx = 0
        test_rmse_, test_rmse_woo_ = [], []
        y_grid_outputs_ = []
        for model_idx in range(len(self.config['epochs'])):
            config = copy.deepcopy(self.config)
            epochs_ran = 0 if model_idx == 0 else self.config['epochs'][model_idx-1]
            config['epochs'] = self.config['epochs'][model_idx] - epochs_ran
            print('==== Epoch %.4f ====' % self.config['epochs'][model_idx])
            self.ft_model.finetune(self.train_jsonl, 
                self.valid_jsonl,
                config,
                saving_checkpoint = log)

            y_valid_outputs = self.query(self.ft_model, valid_prompts, bs = 10, valid_mean = valid_mean)
            y_valid_outputs_.append(y_valid_outputs)
        
            valid_valid_y = [valid_df[y_name][i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]
            valid_valid_y_outputs = [y_valid_outputs[i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]

            len_valid_valid_y = len(valid_valid_y)
            print("| Valid #outputs/Total #outputs:%d/%d" % (len_valid_valid_y,len(y_valid_outputs)))
            len_valid_valid_y_.append(len_valid_valid_y)

            rmse = RMSE(valid_valid_y_outputs, valid_valid_y)
            rmse_woo, num_o = RMSE_woo(valid_valid_y_outputs, valid_valid_y)
            rmse_.append(rmse)
            rmse_woo_.append(rmse)

            print('| RMSE     : %.4f' % rmse)
            print('| RMSE(woo): %.4f   #outlier: %2d}' % (rmse_woo, num_o))
            if (rmse < rmse_[best_idx]) or (np.isnan(rmse_[best_idx])):
                best_idx = model_idx
            
            y_test_outputs = self.query(self.ft_model, test_prompts, bs = 10, valid_mean = valid_mean)
            y_test_outputs_.append(y_test_outputs)

            valid_test_y = [test_df[y_name][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            print("Valid #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test_outputs)))
            test_rmse = RMSE(valid_test_y_outputs, valid_test_y)
            test_rmse_woo, num_o = RMSE_woo(valid_test_y_outputs, valid_test_y)
            test_rmse_.append(test_rmse)
            test_rmse_woo_.append(test_rmse_woo)


            if plot and X_grid is not None and grid_prompts is not None:
                # print('Compute Grid')
                y_grid_outputs = self.query(self.ft_model, grid_prompts, bs = 10, valid_mean = valid_mean)
                # print(y_grid_outputs)
                valid_plot_x = np.array([X_grid[i,0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
                valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
                valid_plot_y_outputs = np.array([y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])

                ax = plt.figure()
                ax.set_facecolor('white')
                
                plt.scatter(valid_plot_x,valid_plot_y_outputs,c=['b']*len(valid_plot_x),label='GPTJ Predicted Labels')
                plt.plot(valid_plot_x,valid_plot_y,c='g',label='True Labels')

                plt.legend()
                plt.title('1D_visualization  n_train='+f'{n_train}'+'\n'\
                        +'Valid #outputs/Total #outputs: '+f'{len(valid_test_y)}'+'/'+f'{len(y_test_outputs)}'+'\n'\
                        +'RMSE      = '+f'{rmse:.3f}'+'\n'\
                        +'RMSE(woo) = '+f'{rmse_woo:.3f}'+'   #outlier: '+f'{num_o}')
                plt.xlabel('x')
                plt.ylabel('y')

                if file_name is None:
                    test_df.to_csv("test_df.csv")
                    plt.savefig('./plot.png',bbox_inches='tight',dpi=300)
                else:
                    try:
                        test_df.to_csv(file_name.split(".")[0]+".csv")
                        plt.savefig(file_name,bbox_inches='tight',dpi=300)
                    except:
                        test_df.to_csv("test_df.csv")
                        plt.savefig('./plot.png',bbox_inches='tight',dpi=300)
            else:
                y_grid_outputs = None
            y_grid_outputs_.append(y_grid_outputs)
                
            if file_name is None:
                test_df.to_csv("test_df.csv")
            else:
                try:
                    test_df.to_csv(file_name.split(".")[0]+".csv")
                except:
                    test_df.to_csv("test_df.csv")
        print('Selected epoch: %.4f' % self.config['epochs'][best_idx])
        self.best_idx = best_idx

        
        return y_test_outputs_[best_idx],y_grid_outputs_[best_idx],len(valid_test_y), test_rmse_[best_idx], test_rmse_woo_[best_idx]

    def log_train(self,num_epochs,train_prompts,valid_prompts,valid_df,test_prompts,test_df,y_name='y', train_df = None, log = False, debug_every = False):
            """
                number of valid samples
                L2 error on the valid samples
            """
            valid_mean = train_df.y.mean()
            loss = defaultdict(list)
            if debug_every:
                loss['train_y'] = train_df[y_name].tolist()
                loss['val_y'] = valid_df[y_name].tolist()
            for epoch in range(num_epochs):
                config = copy.deepcopy(self.config)
                config['epochs'] = 1
                print('==== Epoch %.4f ====' % (epoch+1))
                train_loss, val_loss = self.ft_model.finetune(self.train_jsonl, 
                    self.valid_jsonl,
                    config,
                    saving_checkpoint = log)

                y_train_outputs = self.query(self.ft_model, train_prompts, bs = 10, valid_mean = valid_mean)
                y_valid_outputs = self.query(self.ft_model, valid_prompts, bs = 10, valid_mean = valid_mean)
                y_test_outputs = self.query(self.ft_model, test_prompts, bs = 10, valid_mean = valid_mean)
            
                loss['rmse_train'].append(RMSE(y_train_outputs, train_df[y_name]))
                rmse_woo, _ = RMSE_woo(y_train_outputs, train_df[y_name])
                loss['rmse_woo_train'].append(rmse_woo)
                loss['rae_train'].append(regressionLoss(y_train_outputs, train_df[y_name], 'RAE'))
                rae_woo, _ = regressionLoss(y_train_outputs, train_df[y_name], 'RAE', True)
                loss['rae_woo_train'].append(rae_woo)

                loss['rmse_val'].append(RMSE(y_valid_outputs, valid_df[y_name]))
                rmse_woo, _ = RMSE_woo(y_valid_outputs, valid_df[y_name])
                loss['rmse_woo_val'].append(rmse_woo)
                loss['rae_val'].append(regressionLoss(y_valid_outputs, valid_df[y_name], 'RAE'))
                rae_woo, _ = regressionLoss(y_valid_outputs, valid_df[y_name], 'RAE', True)
                loss['rae_woo_val'].append(rae_woo)

                loss['rmse_test'].append(RMSE(y_test_outputs, test_df[y_name]))
                rmse_woo, _ = RMSE_woo(y_test_outputs, test_df[y_name])
                loss['rmse_woo_test'].append(rmse_woo)
                loss['rae_test'].append(regressionLoss(y_test_outputs, test_df[y_name], 'RAE'))
                rae_woo, _ = regressionLoss(y_test_outputs, test_df[y_name], 'RAE', True)
                loss['rae_woo_test'].append(rae_woo)

                loss['train_loss'].append(train_loss[0])
                loss['val_loss'].append(val_loss[0])

                if debug_every and (epoch % debug_every == 0):
                    loss['train_output_%d' % epoch] = y_train_outputs
                    loss['val_output_%d' % epoch] = y_valid_outputs
            return loss

