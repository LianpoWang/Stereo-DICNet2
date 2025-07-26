



import argparse
import os

import numpy as np
import torch
import time
from UniDIC.network_both import UniDIC
#from UniDIC.network_RAFT import UniDIC
from datasets.SpeckleDataset import SpeckleDataset,Normalization
from loss.loss import flow_loss_func_test, disp_loss_func_test,flow_loss_func_test_test
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CUDA_VISIBLE_DEVICES=2
device_ids = [0,1]
parser = argparse.ArgumentParser(description='ACVNet')

parser.add_argument('--loadmodel', default='result/best_val_model.ckpt',
                    help='loading model')  #加载预训练模型  
parser.add_argument('--model', default='UniDIC',     
                    help='select model')   #加载训练

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)

transform = transforms.Compose([Normalization()])



test_data = SpeckleDataset(csv_file='xxxxxxxxxxxxxxxx.csv',
                            root_dir='/xxxxxxxxxxx', transform=transform)

test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=2,
        num_workers=1, pin_memory=True, shuffle=False)

# model
feature_channels = 128
num_scales = 1
upsample_factor = 8
num_head = 1
ffn_dim_expansion = 4
num_transformer_layers = 6
reg_refine = True
task = "flow"

model = UniDIC(feature_channels=feature_channels,
                     num_scales=num_scales,
                     upsample_factor=upsample_factor,
                     num_head=num_head,
                     ffn_dim_expansion=ffn_dim_expansion,
                     num_transformer_layers=num_transformer_layers,
                     reg_refine=reg_refine,
                     task=task).to(device), #加载网络结构  
model =model[0]
model = torch.nn.DataParallel(model, device_ids=device_ids)
model = model.module

model.eval()
model.training=False


if args.loadmodel is not None:
    print('load UniDIC')
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    state_dict = torch.load(args.loadmodel)
    #print(state_dict)
    model.load_state_dict(state_dict['model'], strict=False)  # 加载模型参数
    for name, param in model.named_parameters():
        print(name)
# state_dict = torch.load(args.loadmodel)
# model.load_state_dict(state_dict['model'],strict=False)




def test(test_loader, model):
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
        total_test_loss = 0
        avg_flow_x_loss = 0
        avg_flow_y_loss = 0
        avg_disp_loss = 0
        test_losses_flow = []
        test_losses_disp = []
        count=0
        
        for batch_idx, sample in enumerate(test_loader):

            disp_gt = sample['Dispx'].to(device)  # n*1*256*256
            u = sample['U'].to(device).unsqueeze(1)
            v = sample['V'].to(device).unsqueeze(1)
            flow_gt = torch.cat([u, v], 1).to(device)
            L0 = sample['L0'].float().to(device)
            R0 = sample['R0'].float().to(device)
            L1 = sample['L1'].float().to(device)
            start_time = time.time()

           

            results_dict = model(L0, R0, L1)
            
            flow_preds = results_dict[0]
            disp_preds = results_dict[1]

            loss_flow_x,loss_flow_y = flow_loss_func_test_test(flow_preds, flow_gt)
            loss_flow= flow_loss_func_test(flow_preds, flow_gt)
            loss_disp = disp_loss_func_test(disp_preds, disp_gt)
            test_losses_flow.append(loss_flow.cpu().detach().numpy())
            test_losses_disp.append(loss_disp.cpu().detach().numpy())

            loss = 0.6 * loss_flow + 0.6 * loss_disp
            # loss =  loss_flow

            total_test_loss += loss
            avg_flow_x_loss +=loss_flow_x
            avg_flow_y_loss +=loss_flow_y
            avg_disp_loss +=loss_disp

        #     print('Test: [{0}/{1}]\t Time {2}\t loss_flow {3}'
        #           .format(batch_idx, len(test_loader), (time.time() - start_time), loss_flow))
            print('Test: [{0}/{1}]\t Time {2}\t loss_flow_x {3}\t loss_flow_y {4}\t loss_disp {5}'
                  .format(batch_idx, len(test_loader), (time.time() - start_time), loss_flow_x, loss_flow_y, loss_disp))
        print('total_time:',(time.time() - start_time),'avg_flow_x_loss:',avg_flow_x_loss/len(test_loader),'avg_flow_y_loss:',avg_flow_y_loss/len(test_loader),'avg_disp_loss:',avg_disp_loss/len(test_loader))
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(test_losses_flow)), test_losses_flow, label="test loss_flow")
        plt.xlabel('test example')
        plt.ylabel("Loss_flow")
        plt.title('Model loss flow')
        plt.show()

        plt.plot(np.arange(len(test_losses_disp)), test_losses_disp, label="test loss_disp")
        plt.legend()  # 显示图例
        plt.xlabel('test example')
        plt.ylabel("Loss_disp")
        plt.title('Model loss disp')
        plt.show()





def main():
        test(test_loader, model)

if __name__ == '__main__':
   main()






