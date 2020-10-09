# -*- coding: utf-8 -*-

class HuberLoss(torch.nn.Module):
    
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_predict, y):
        residual = torch.abs(torch.sub(y, y_predict))
        mask1 = residual <= self.delta
        lossMSE = torch.sum(0.5 * torch.pow(residual[mask1], 2))
        mask2 = residual > self.delta
        lossMAE = torch.sum(torch.sub(self.delta * residual[mask2], 0.5 * self.delta**2))
        
        #return torch.div(torch.add(lossMSE, lossMAE), y.size()[0])
        return (lossMSE + lossMAE) / (y.size()[0])



"""
Example:
criterion = torch.nn.MSELoss()
criterion = torch.nn.SmoothL1Loss()
criterion = HuberLoss(delta=1)  
"""

"""
class HuberLoss(torch.nn.Module):
    
    def __init__(self, delta):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_predict, y):
        residual = abs(y - y_predict)
        mask1 = residual <= self.delta
        lossMSE = sum(0.5 * residual[mask1]**2)
        mask2 = residual > self.delta
        lossMAE = sum(self.delta * residual[mask2] - 0.5 * self.delta**2)
        
        return (lossMSE + lossMAE) / (y.size()[0])
"""

