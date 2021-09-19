import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_with_dropout(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_with_dropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.3)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, pretrained, method, dataset):
        super(ResNet, self).__init__()

        self.method = method
        self.dataset = dataset
        self.pretrained = pretrained
        ## self defined resnet
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)


        ## pretrained network
        self.resnet =  models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
        self.intermediate = {}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_intermediate(self, name):
        def hook(model, input, output):
            self.intermediate[name] = output.detach()
        return hook

    def forward(self, x):
        
        if self.pretrained == False:
            ## self defined resnet
            out = F.relu(self.bn1(self.conv1(x)))
            out1 = self.layer1(out)       
            out2 = self.layer2(out1)     
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out = self.avgpool(out4)
            representation = out.view(out.size(0), -1)
            out = self.linear(representation)
            
            #print("out1 size: ", out1.shape)
            #print("out2 size: ", out2.shape)
            #print("out3 size: ", out3.shape)
            #print("out4 size: ", out4.shape)

            if self.method == "Coreset":
                return representation, out    
            elif self.method == "LLAL":
                return out, [out1, out2, out3, out4]
            else:
                return out
        else:
            ## using pretrained network
            
            # setup the hook for the intermediate layer
            out1_hook = self.resnet.layer1.register_forward_hook(self.get_intermediate('layer1'))
            out2_hook = self.resnet.layer2.register_forward_hook(self.get_intermediate('layer2'))
            out3_hook = self.resnet.layer3.register_forward_hook(self.get_intermediate('layer3'))
            out4_hook = self.resnet.layer4.register_forward_hook(self.get_intermediate('layer4'))
            representation_hook = self.resnet.avgpool.register_forward_hook(self.get_intermediate('avgpool'))

            #print(self.resnet)

            out = self.resnet(x) 
   
            
            if self.method == "Coreset":
                representation = self.intermediate['avgpool']
                print("representation: ", representation)
                return representation, out
            elif self.method == "LLAL":
                out1 = self.intermediate['layer1']
                out2 = self.intermediate['layer2']
                out3 = self.intermediate['layer3']
                out4 = self.intermediate['layer4']
                #print("out1 size: ", out1.shape)
                #print("out2 size: ", out2.shape)
                #print("out3 size: ", out3.shape)
                #print("out4 size: ", out4.shape)

                return out, [out1, out2, out3, out4]
            else:
                return out
            
            out1_hook.remove()
            out2_hook.remove()
            out3_hook.remove()
            out4_hook.remove()
            representation_hook.remove()

def ResNet18(num_classes, pretrained, method, dataset):
    print("num_classes:", num_classes)
    print("Pretrained: ", pretrained)
    print("method: ", method)
    resnet18 = ResNet(BasicBlock, [2,2,2,2], num_classes, pretrained, method, dataset)
    return resnet18


def ResNet18_with_dropout(num_classes):
    return ResNet(BasicBlock_with_dropout, [2,2,2,2], num_classes, method, dataset)



def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
