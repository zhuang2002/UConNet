import argparse
import torchvision as tv
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from lib.dataset import *
from lib.utils import *
from Network.model import *
import cv2 as cv
# from Net_test import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from matplotlib import pyplot as plt
transform = tv.transforms.Compose(
    [#tv.transforms.Resize(128),
        tv.transforms.ToTensor()
        # tv.transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])
seed=1#424
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
parser = argparse.ArgumentParser(description="AngleNet")
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
parser.add_argument("--milestone", type=list, default=[30, 60, 80], help="When to decay learning rate;")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="output", help='path of log files')
parser.add_argument("--rain_path",type=str,default=r" ",help='path of rain dataset')
parser.add_argument("--real_path",type=str,default=r" ",help='path of clean path')
parser.add_argument("--reset", type=int, default=1, help='path of dataset')
opt = parser.parse_args()

def main():
    repeat = 5700
    device_ids = [0]
    dataset_train = Angle_data(opt.rain_path,transforms=transform,patch_size=128,batch_size=opt.batchSize,repeat=repeat,channel=3)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True)


    AngleNet = Angle_Net(in_channels=3).cuda()
    AngleNet.apply(weights_init_kaiming)
    AngleNet = nn.DataParallel(AngleNet, device_ids=device_ids).cuda()
    angle_optimizer = torch.optim.Adam(AngleNet.parameters(), lr=opt.lr, eps=1e-4, amsgrad=True)
    angle_schedulr = torch.optim.lr_scheduler.MultiStepLR(angle_optimizer, milestones=[30, 40, 50], gamma=0.2, last_epoch=-1)

    F_normal = nn.MSELoss()

    for epoch in range(opt.epochs):
        mean_loss = 0
        for i, data in  enumerate(loader_train):
            rain2, angle_label = data['rain_data'].type(torch.FloatTensor).cuda(), \
                                             data['angle_label'].type(torch.FloatTensor).cuda()
            # print(angle_label*90-45)
            out_angle = AngleNet(rain2)
            angle_loss = F_normal(out_angle,angle_label)
            mean_loss +=angle_loss
            angle_optimizer.zero_grad()
            angle_loss.backward()
            angle_optimizer.step()

        print('The '+str(epoch)+' epoch',mean_loss/5700)
        if epoch%1==0:
            torch.save(AngleNet.state_dict(), r'.\result_model\AngleNet/AngleNet_epoch'+str(epoch)+'.pth')

        angle_schedulr.step()



if __name__ == "__main__":
    main()
