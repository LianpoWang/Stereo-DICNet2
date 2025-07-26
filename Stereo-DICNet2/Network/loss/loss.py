import torch
import torch.nn.functional as F

def flow_loss_func(flow_preds, flow_gt,gamma=0.9):

    n_predictions = len(flow_preds)  #5


    flow_loss = 0.0
    flow_loss_x = 0.0
    flow_loss_y = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        # i_loss = (flow_preds[i] - flow_gt).abs()
        # flow_loss += i_weight * i_loss.mean()

        # flow_loss += i_weight * F.l1_loss(flow_preds[i],flow_gt)
        # print(flow_preds[i][0, :, :].size())
        # print(flow_gt[0, :, :].size())

        flow_loss_x += i_weight * F.smooth_l1_loss(flow_preds[i][0, :, :], flow_gt[0, :, :]).mean()
        flow_loss_y += i_weight * F.smooth_l1_loss(flow_preds[i][1, :, :], flow_gt[1, :, :]).mean()
        # flow_loss = 0.8 * flow_loss_x + 0.2 * flow_loss_y
        #flow_loss = (flow_loss_x + 0.6 * flow_loss_y) / 2
        flow_loss = flow_loss_x 


    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()



    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss


def disp_loss_func(flow_preds, flow_gt,gamma=0.9):


    n_predictions = len(flow_preds)
    flow_loss = 0.0
   
    for i in range(n_predictions):
        #print('n_predictions:',n_predictions) 
        i_weight = gamma ** (n_predictions - i - 1)

        # i_loss = (flow_preds[i] - flow_gt).abs()
        # flow_loss +=  i_loss
        # flow_preds[i] = flow_preds[i]
        # flow_preds[i] = flow_preds[i][:,0,:,:]
        # print(' disp_pred:', flow_preds[i].size())  #[20, 1, 256, 256]
        # print(' flow_gt:', flow_gt.size()) #[20, 1, 256, 256]
        flow_loss += i_weight * F.smooth_l1_loss(flow_preds[i], flow_gt).mean()
     

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()



    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss


def flow_loss_func_test(flow_preds, flow_gt,gamma=0.9):

    n_predictions = len(flow_preds)
    

    flow_loss = 0.0
    flow_loss_x = 0.0
    flow_loss_y = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        # i_loss = (flow_preds[i] - flow_gt).abs()
        # flow_loss += i_weight * i_loss.mean()

        # flow_loss += i_weight * F.l1_loss(flow_preds[i],flow_gt)
        flow_loss_x += i_weight * F.smooth_l1_loss(flow_preds[i][:,0, :, :], flow_gt[:,0, :, :]).mean()
        flow_loss_y += i_weight * F.smooth_l1_loss(flow_preds[i][:,1, :, :], flow_gt[:,1, :, :]).mean()
        # flow_loss = 0.8 * flow_loss_x + 0.2 * flow_loss_y
        # print('flow_loss_x:',flow_loss_x)
        # print('flow_loss_y:',flow_loss_y)
        #flow_loss = (flow_loss_x + 0.8 * flow_loss_y)/2
        flow_loss = flow_loss_x

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()



    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss


def flow_loss_func_test_test(flow_preds, flow_gt,gamma=0.9):

    n_predictions = len(flow_preds)

    flow_loss = 0.0
    flow_loss_x = 0.0
    flow_loss_y = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

        # i_loss = (flow_preds[i] - flow_gt).abs()
        # flow_loss += i_weight * i_loss.mean()

        # flow_loss += i_weight * F.l1_loss(flow_preds[i],flow_gt)
        # flow_loss_x += i_weight * F.smooth_l1_loss(flow_preds[i][:,0, :, :], flow_gt[:,0, :, :]).mean()
        # flow_loss_y += i_weight * F.smooth_l1_loss(flow_preds[i][:,1, :, :], flow_gt[:,1, :, :]).mean()
        flow_loss_x += F.l1_loss(flow_preds[i][:,0, :, :], flow_gt[:,0, :, :])
        flow_loss_y += F.l1_loss(flow_preds[i][:,1, :, :], flow_gt[:,1, :, :])
        # flow_loss = 0.8 * flow_loss_x + 0.2 * flow_loss_y
        flow_loss = (flow_loss_x + 0.8 * flow_loss_y)/2

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()



    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return flow_loss_x, flow_loss_y


def disp_loss_func_test(flow_preds, flow_gt,gamma=0.9):


    n_predictions = len(flow_preds)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)

       
        flow_loss += i_weight * F.l1_loss(flow_preds[i], flow_gt).mean()

    # epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    #
    #
    #
    # metrics = {
    #     'epe': epe.mean().item(),
    #     '1px': (epe > 1).float().mean().item(),
    #     '3px': (epe > 3).float().mean().item(),
    #     '5px': (epe > 5).float().mean().item(),
    # }

    return flow_loss