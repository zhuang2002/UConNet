import torch
import numpy as np
from Network.model import *
import torchvision as tv
from lib.dataset import *
import argparse
from torch.utils.data import DataLoader
seed=1#424
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
parser = argparse.ArgumentParser(description="AngleNet")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
parser.add_argument("--milestone", type=list, default=[20, 60, 80], help="When to decay learning rate;")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="output", help='path of log files')
# parser.add_argument("--rain_path",type=str,default=r"E:\zjh_code\train\rain",help='path of rain dataset')
parser.add_argument("--rain_path",type=str,default=r"",help='path of rain dataset')
# parser.add_argument("--clean_path",type=str,default=r"E:\zjh_code\train\gt",help='path of clean path')
parser.add_argument("--clean_path",type=str,default=r"",help='path of clean path')
parser.add_argument("--derainNet_path",type=str,default=r'.\result_model/UConNet/UconNet_2.pth',help='path of clean path')
parser.add_argument("--AngleNet_path",type=str,default=r'.\result_model\AngleNet/AngleNet.pth',help='path of clean path')
parser.add_argument("--reset", type=int, default=1, help='path of dataset')
parser.add_argument("--root",type=str,default=r"",help='path of rain dataset')
opt = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from matplotlib import pyplot as plt
transform = tv.transforms.Compose(
    [#tv.transforms.Resize(128),
        tv.transforms.ToTensor()
        # tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def main():
    repeat = 2500
    device_ids = [0]
    dataset_train = param_data2(opt.rain_path,opt.clean_path, transforms=transform, patch_size=128, batch_size=opt.batchSize,repeat=repeat, channel=3)

    loader_train = DataLoader(dataset=dataset_train, num_workers=16, batch_size=opt.batchSize, shuffle=True)
    derainNet = derain_Net1().cuda()
    derainNet = nn.DataParallel(derainNet).cuda()
    derainNet.load_state_dict(torch.load(opt.derainNet_path))
    set_requires_grad(derainNet,False)

    paramNet = AdjParam_Net().cuda()
    paramNet.apply(weights_init_kaiming)
    param_optimizer = torch.optim.Adam(paramNet.parameters(), lr=opt.lr, eps=1e-4, amsgrad=True)
    paramNet = nn.DataParallel(paramNet, device_ids=device_ids).cuda()
    # paramNet.load_state_dict(torch.load(r'result_model/param_net/param_net_Dataset00change_'+str(10)+'.pth'))
    param_schedulr = torch.optim.lr_scheduler.MultiStepLR(param_optimizer, milestones=[10, 20, 40], gamma=0.2,
                                                           last_epoch=-1)

    AngleNet = Angle_Net(in_channels=3).cuda()
    AngleNet = nn.DataParallel(AngleNet).cuda()
    AngleNet.load_state_dict(torch.load(r'.\result_model\AngleNet/AngleNet.pth'))
    AngleNet.eval()


    F_loss = torch.nn.MSELoss()
    # ssim_loss  = SSIM()
    for epoch in range(opt.epochs):
        mean_loss = 0
        for i, data in enumerate(loader_train):
            image,clean_image = data['rain_data'].type(torch.FloatTensor).cuda(),data['clean_data'].type(torch.FloatTensor).cuda()

            image_Y,Cb,Cr = yCbCr2rgb(image)
            image_Y = image_Y.unsqueeze(1)
            clean_y,Cb,Cr = yCbCr2rgb(clean_image)
            clean_y = clean_y.unsqueeze(1)
            with torch.no_grad():
                angle_predicted = AngleNet(image)
                angle_predicted = (angle_predicted*120-60)/180*3.1415926535897
                angle_predicted = torch.reshape(angle_predicted,(angle_predicted.shape[0],1))

                out1_temp,out2_temp,out3_temp = derainNet(image_Y.detach(),None,angle_predicted.detach(),None,None,None,False)
                input_feature = torch.cat((out1_temp,out2_temp,out3_temp),dim=1).cuda()
                # print(input_feature.shape)

            a = paramNet(input_feature.detach()).permute(1,0)

            out = derainNet(image_Y.detach(),a,angle_predicted.detach(),out1_temp,out2_temp,out3_temp,True)
            # out = derainNet(image_Y.detach(),a,angle_predicted.detach(),True)
            loss = F_loss(out,clean_y)
            # print(clean_y)

            mean_loss+=loss
            param_optimizer.zero_grad()
            loss.backward()
            param_optimizer.step()
        print('The ' + str(epoch) + ' epoch:', mean_loss )
        if epoch % 1 == 0:
            torch.save(paramNet.state_dict(), r'.\result_model\ParamNet\ParamNet_epoch' + str(epoch) + '.pth')
            # test_net('Angle_net_withReal_'+str(epoch)+'.pth')
        param_schedulr.step()

if __name__ == "__main__":
    main()