import time
import pprint
import sys
import IPython
import mir_eval
import numpy as np
import tensorflow as tf
import scipy.io
tf.reset_default_graph()
sys.path.append('../')
from model.DataProvider import DataProvider
from model.EarlyStopping import EarlyStopping
from model.NetSaver import NetSaver
from model.UNet import UNet
from model import Loss
from model import Trainer
from model.DataArgument import DataArgument
from model import Masks
from model.STFT_Module import STFT_Module
from lib import AudioModule
from visualizer import visualize_loss
from visualizer import visualize_spec


class Test():
        def __init__(self, epoch_num=1, batch_size=10, fs = 16000, sec = 18, test_data_num=10000, sample_len=65536):
                self.epoch_num= epoch_num
                self.batch_size = batch_size
                self.input_shape = None
                self.test_iter = None
                self.fs = fs
                self.sec = sec
                self.test_data_num = test_data_num
                self.est_audio_list = []
                self.sdr_list = []
                self.sir_list = []
                self.sar_list = []
                self.sample_len = sample_len
                self.epsilon = 1e-4
                self.min_frame_step = 128
                self.max_fft_length = 2048
                self.stft_params = {
                        "frame_length": 1024,
                        "frame_step":  self.min_frame_step,
                        "fft_length": self.max_fft_length,
                        "pad_end": True
                }
        
        def expand_channel(self, tf_X):
                return tf.expand_dims(tf_X, -1)
            
        def __model(self, tf_mix):
                 # define model flow
                # stft
                stft_module = STFT_Module(
                        frame_length = self.stft_params["frame_length"], 
                        frame_step= self.stft_params["frame_step"], 
                        fft_length = self.stft_params["fft_length"],
                        epsilon = self.epsilon,
                        pad_end = self.stft_params["pad_end"]
                )
                
                # mix data transform
                tf_spec_mix = stft_module.STFT(tf_mix)
                tf_phase_mix = tf.sign(tf_spec_mix)
                tf_phase_mix = self.expand_channel(tf_phase_mix)

                tf_amp_spec_mix = stft_module.to_amp_spec(tf_spec_mix, normalize =False)
                tf_mag_spec_mix = tf.log(tf_amp_spec_mix + self.epsilon)
                tf_mag_spec_mix = tf.expand_dims(tf_mag_spec_mix, -1)# (Batch, Time, Freq, Channel))
                tf_amp_spec_mix = tf.expand_dims(tf_amp_spec_mix, -1)
                tf_mag_spec_mix = tf_mag_spec_mix[:,:,:1024,:]

                u_net = UNet(
                        input_shape =(
                             tf_mag_spec_mix.shape[1:]
                        )
                )
            
                tf_est_masks = u_net(tf_mag_spec_mix)
                
                zero_pad = tf.zeros_like(tf_mag_spec_mix)
                zero_pad = tf.expand_dims(zero_pad[:,:,1,:], -1)
                tf_est_masks = tf.concat([zero_pad, tf_est_masks], 2)
                tf_est_spec = tf.math.multiply(tf_est_masks, tf_amp_spec_mix)
                tf_est_source_spec = tf.math.multiply(tf.complex(tf_est_spec, 0.), tf_phase_mix)
                tf_est_source_spec = tf.squeeze(tf_est_source_spec, axis=-1)                
                est_source = stft_module.ISTFT(tf_est_source_spec)
                return est_source
                
        def __call__(self):                                              
                # load all train data
                provider = DataProvider()
                test_bass_list, test_drums_list, test_other_list, test_vocals_list = provider.load_all_test_data()
                # define model
                tf_mix = tf.placeholder(tf.float32, (None, self.sample_len)) #Batch, Sample
                tf_est_source = self.__model(tf_mix)
                
                # GPU config
                config = tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                                visible_device_list='0', # specify GPU number
                                allow_growth = True
                        )
                )
                
                saver = tf.train.import_meta_graph('./../results/model/u_net/u_net_1980.ckpt.meta')
                with tf.Session(config = config) as sess:
                        saver.restore(sess, './../results/model/u_net/u_net_1980.ckpt')
                        
                        test_mixed_list = []
                        for bass, drums, other, vocals in zip(test_bass_list, test_drums_list, test_other_list, test_vocals_list):
                            test_mixed_list.append(AudioModule.mixing(
                                                                                bass,
                                                                                drums,
                                                                                other,
                                                                                vocals,
                                                               ))
                        test_target_list = test_vocals_list
                        tf.keras.backend.set_learning_phase(0)
                        # make mix audio
                        est_start = time.time()
                        for mix in test_mixed_list:
                            cutted_mix_array = provider.test_data_split_and_pad(mix, self.sample_len)                            
                            tmp_est_data_array = np.zeros((len(cutted_mix_array), self.sample_len))
                            for index, mix_packet in enumerate(cutted_mix_array):
                                mix_packet = mix_packet.reshape(1,-1)
                                est_source = sess.run(tf_est_source, feed_dict = {
                                       tf_mix: mix_packet[:,:]
                                    }
                                 )
                                tmp_est_data_array[index,:] = est_source[:, :self.sample_len]
                                
                            self.est_audio_list.append(tmp_est_data_array.reshape(1,-1))
                        est_end = time.time()
                        print("excuted time", est_end - est_start)
                        
                        evaluate_start = time.time()
                        for est, target, mix in zip(self.est_audio_list, test_target_list, test_mixed_list):
                                target = target.reshape(1,-1)
                                mix = mix.reshape(1,-1)
                                
                                est_array = np.zeros((2, target.shape[1]))
                                est_array[0,:] = est[:, :target.shape[1]]
                                est_array[1,:] = mix[:, :target.shape[1]] - est[:, :target.shape[1]]

                                target_array = np.zeros((2, target.shape[1]))
                                target_array[0,:] = target
                                target_array[1,:] = mix[:, :target.shape[1]] - target
                                
                                sdr, sir, sar, perm =  mir_eval.separation.bss_eval_sources(target_array, est_array)
                                self.sdr_list.append(sdr[0])
                                self.sir_list.append(sir[0])
                                self.sar_list.append(sar[0])
                        print('sdr mean',np.mean(self.sdr_list))
                        print('sir mean',np.mean(self.sir_list))
                        print('sar mean',np.mean(self.sar_list))
                        
                        print('sdr median', np.median(self.sdr_list))
                        print('sir median', np.median(self.sir_list))
                        print('sar median', np.median(self.sar_list))
                        
                        evaluate_end = time.time()
                        print('evaluate time', evaluate_end - evaluate_start)
                return self.est_audio_list,  test_target_list, test_mixed_list, self.sdr_list, self.sir_list, self.sar_list 
                
if __name__ == '__main__':
    test = Test()
    est_list, target_list, mixed_list, sdr_list, sir_list, sar_list = test()
#    scipy.io.savemat("./../results/mat/u_net_sdr.mat", {'u_net_sdr_list':sdr_list})
#    scipy.io.savemat("./../results/mat/u_net_sir.mat", {'u_net_sir_list':sir_list})
#    scipy.io.savemat("./../results/mat/u_net_sar.mat", {'u_net_sar_list':sar_list})
    file_path = './../results/audio/UNet/singing_voice_separation/'
     #AudioModule.to_pickle(est_list, file_path + 'est_list')
   