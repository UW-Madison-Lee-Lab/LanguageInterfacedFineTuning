import openai, os, time
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

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

class GPT3FineTuner(object):
    def __init__(self,config:dict,train_jsonl,valid_jsonl,openai_key='[REPLACE IT WITH YOUR OPENAI KEY]'):
        self.config = config
        self.train_jsonl=train_jsonl
        self.valid_jsonl=valid_jsonl
        
        self.file_info = openai.File.create(file = open(train_jsonl), purpose = 'fine-tune')
        self.training_file_id   = self.file_info['id']
        self.file_info = openai.File.create(file = open(valid_jsonl), purpose = 'fine-tune')
        self.validation_file_id   = self.file_info['id']

        self.openai_key = openai_key

    
    def init_model(self):
        self.fine_tuned = False
        self.ft_info = [openai.FineTune.create(training_file = self.training_file_id,
                                 validation_file = self.validation_file_id,
                                 model = self.config['model_type'],
                                 n_epochs = self.config['num_epochs'],
                                 batch_size = self.config['batch_size'],
                                 learning_rate_multiplier = learning_rate_multiplier,
                                 #prompt_loss_weight = prompt_loss_weight,
                                 #compute_classification_metrics = compute_classification_metrics,
                                 #classification_n_classes = classification_n_classes,
                                 #classification_positive_class = classification_positive_class,
                                 #classification_betas = classification_betas
                                 ) for learning_rate_multiplier in self.config['lr']]

    def fine_tune(self):
        self.init_model()
    
    def query(self,prompt,model,valid_temperature=0.75,valid_mean = 0):
        load_flag = True
        while(load_flag):
            try:
                output =  openai.Completion.create(model = model,prompt = prompt, temperature=0)['choices'][0]['text']
                load_flag = False
            except Exception as e:
                print("%s" % e)
                load_flag = True
                time.sleep(10)
        try:
            return float(output.split('@@@')[0])
        except:
            load_flag = False
            for _ in range(5):
                try:
                    output =  openai.Completion.create(model = model,prompt = prompt, temperature=valid_temperature)['choices'][0]['text']
                    load_flag = False
                except Exception as e:
                    print("%s" % e)
                    load_flag = True
                    time.sleep(10)
                
                try:
                    return float(output.split('@@@')[0])
                except:
                    pass
        return valid_mean



    def eval(self,n_train,valid_prompts,valid_df,test_prompts,test_df,training_csv_file_name,y_name='y',plot=False,X_grid=None,grid_prompts=None,y_grid=None,file_name=None, train_df = None, valid_temperature = 0.75):
        """
            number of valid samples
            L2 error on the valid samples
        """
        valid_mean = train_df[y_name].mean()
        y_valid_outputs_,len_valid_valid_y_, rmse_, rmse_woo_ = [], [], [], []
        best_idx = 0
        self.ft_model, self.ft_id = [],[]
        for model_idx in range(len(self.config['lr'])):
            print('==== Learning rate multiplier %.4f ====' % self.config['lr'][model_idx])
            self.ft_id.append(self.ft_info[model_idx]['id'])
            self.finetune_status = None
            while(self.finetune_status != 'succeeded'):
                self.ft_info[model_idx] = openai.FineTune.retrieve(id=self.ft_id[model_idx])
                time.sleep(10)
                if self.finetune_status != self.ft_info[model_idx]['status']:
                    self.finetune_status = self.ft_info[model_idx]['status']
                    print("| %s " % self.finetune_status)
                if self.finetune_status == 'failed':
                    print("| Recreate a new finetuning task!")
                    self.ft_info[model_idx] = openai.FineTune.create(training_file = self.training_file_id,
                                 validation_file = self.validation_file_id,
                                 model = self.config['model_type'],
                                 n_epochs = self.config['num_epochs'],
                                 batch_size = self.config['batch_size'],
                                 learning_rate_multiplier = self.config['lr'][model_idx],
                                 #prompt_loss_weight = prompt_loss_weight,
                                 #compute_classification_metrics = compute_classification_metrics,
                                 #classification_n_classes = classification_n_classes,
                                 #classification_positive_class = classification_positive_class,
                                 #classification_betas = classification_betas
                                 )
                    self.ft_id[model_idx] = self.ft_info[model_idx]['id']
            self.ft_model.append(self.ft_info[model_idx]['fine_tuned_model'])
            print('| fine-tune id: ',self.ft_id[model_idx])
            print('| fine-tune model: ',self.ft_info[model_idx]['fine_tuned_model'])

            y_valid_outputs = list(map(partial(self.query, model = self.ft_model[model_idx], valid_mean = valid_mean, valid_temperature = valid_temperature), valid_prompts))
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
        print('Selected learning rate: %.4f' % self.config['lr'][best_idx])
        os.system("""export OPENAI_API_KEY="%s"
        openai api fine_tunes.results -i %s > %s""" % (self.openai_key, self.ft_id[best_idx], training_csv_file_name))

        y_test_outputs = list(map(partial(self.query, model = self.ft_model[best_idx], valid_mean = valid_mean, valid_temperature = valid_temperature),test_prompts))
        valid_test_y = [test_df[y_name][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
        valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
        print("Valid #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test_outputs)))
        rmse = RMSE(valid_test_y_outputs, valid_test_y)
        rmse_woo, num_o = RMSE_woo(valid_test_y_outputs, valid_test_y)
        self.ft_info = self.ft_info[best_idx]
        self.ft_id = self.ft_id[best_idx]
        self.ft_model = self.ft_model[best_idx]
        self.best_idx = best_idx

        if plot and X_grid is not None and grid_prompts is not None:
            
            #print(grid_prompts)
            y_grid_outputs = list(map(partial(self.query, model = self.ft_model, valid_mean = valid_mean, valid_temperature = valid_temperature),grid_prompts))
            valid_plot_x = np.array([X_grid[i,0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
            valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
            valid_plot_y_outputs = np.array([y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])

            ax = plt.figure()
            ax.set_facecolor('white')
            
            plt.scatter(valid_plot_x,valid_plot_y_outputs,c=['b']*len(valid_plot_x),label='GPT3 Predicted Labels')
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
            
        if file_name is None:
            test_df.to_csv("test_df.csv")
        else:
            try:
                test_df.to_csv(file_name.split(".")[0]+".csv")
            except:
                test_df.to_csv("test_df.csv")
        
        return y_test_outputs,y_grid_outputs,len(valid_test_y), rmse, rmse_woo
