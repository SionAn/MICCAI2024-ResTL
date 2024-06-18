import torch

def one_epoch_distribution(net, data_loader, optimizer, loss, hyper, args, train='train'):
    acc_epoch = 0
    loss_epoch = 0
    count_epoch = 0
    loss_print = {'Total': 0, 'CLS': 0, 'Dist': 0, 'Sub': 0, 'Sample': 0}
    epoch_dist = {'mu': [], 'sigma': []}
    net.mu.requires_grad = False
    net.sigma.requires_grad = False
    for i_iter, sample_train in enumerate(data_loader):
        x_anc, x_pos, x_neg, label, x_anc_rest, x_pos_rest, x_neg_rest = sample_train.values()
        if train == 'train' or train == 'val':
            x = torch.cat([x_anc, x_pos, x_neg, x_anc_rest, x_pos_rest, x_neg_rest], dim=0).to(args.device)
        else:
            x = x_anc.to(args.device)
        y = sample_train['label'].argmax(dim=1).type(torch.LongTensor).to(args.device)
        output, task, subject = net.forward(x)

        # classification
        output, _ = torch.split(output, [x_anc.shape[0], x.shape[0]-x_anc.shape[0]], dim=0)
        if train != 'RS':
            clsloss = loss(output, y)
        else:
            y_ = sample_train['label'].to(args.device)
            clsloss = loss(torch.nn.functional.log_softmax(output, dim=1), y_)

        # task
        task_anc, _ = torch.split(task, [x_anc.shape[0], x.shape[0] - x_anc.shape[0]], dim=0)
        mu = task_anc.mean(dim=(2, 3))
        sigma = task_anc.std(dim=(2, 3))
        dist_mu = []
        dist_sigma = []
        metric_mu = []
        cdist_mu = torch.cdist(mu, net.mu.detach())
        cdist_sigma = torch.cdist(sigma, net.sigma.detach())
        for i in range(args.datanum):
            idx = torch.where(y == i)[0]
            if idx.shape[0] != 0:
                if args.multiproto == 1:
                    dist_mu.append(cdist_mu[i, idx, 0].mean())
                    dist_sigma.append(cdist_sigma[i, idx, 0].mean())
                    for j in range(1, args.datanum):
                        metric_mu.append(torch.max(torch.zeros(1).to(args.device), cdist_mu[i, idx, 0] - cdist_mu[i - j, idx, 0] + 1))

        muloss = torch.stack(dist_mu)
        epoch_dist['mu'].append(muloss)
        sigmaloss = torch.stack(dist_sigma)
        epoch_dist['sigma'].append(sigmaloss)
        distloss = muloss.mean() + torch.cat(metric_mu).mean()

        # subject
        if subject.dim() == 2:
            subject = subject.unsqueeze(2).unsqueeze(3)
        if train == 'train' or train == 'val':
            sub_anc, sub_pos, sub_neg, sub_anc_rest, sub_pos_rest, sub_neg_rest = (
                torch.split(subject, [x_anc.shape[0], x_anc.shape[0], x_anc.shape[0], x_anc.shape[0], x_anc.shape[0], x_anc.shape[0]]))
            pdist = torch.nn.PairwiseDistance()
            d_pos = pdist(sub_anc.mean(dim=(2, 3)), sub_pos.mean(dim=(2, 3))).mean()
            d_neg = pdist(sub_anc.mean(dim=(2, 3)), sub_neg.mean(dim=(2, 3))).mean()
            subjectloss1 = torch.max(torch.zeros(1).to(args.device), (d_pos - d_neg + 1)).mean() #+ d_pos
            d_pos_rest = pdist(sub_anc_rest.mean(dim=(2, 3)), sub_pos_rest.mean(dim=(2, 3))).mean()
            d_neg_rest = pdist(sub_anc_rest.mean(dim=(2, 3)), sub_neg_rest.mean(dim=(2, 3))).mean()
            subjectloss2 = torch.max(torch.zeros(1).to(args.device), (d_pos_rest - d_neg_rest + 1)).mean() #+ d_pos_rest
            d_pos_rest_mix = pdist(sub_anc.mean(dim=(2, 3)), sub_pos_rest.mean(dim=(2, 3))).mean()
            d_neg_rest_mix = pdist(sub_anc.mean(dim=(2, 3)), sub_neg_rest.mean(dim=(2, 3))).mean()
            subjectloss3 = torch.max(torch.zeros(1).to(args.device), (d_pos_rest_mix - d_neg_rest_mix + 1)).mean() #+ d_pos_rest_mix
            subjectloss = (subjectloss1 + subjectloss2 + subjectloss3)/3
        else:
            subjectloss = torch.zeros(1).to(args.device)

        # rest
        if train == 'train' or train == 'val':
            sampled_x = []
            sampled_y = []
            if task_anc.shape[0] != 1:
                sample_shape = [task_anc.shape[0]//args.datanum, task_anc.shape[2], task_anc.shape[3]]
            else:
                sample_shape = [1, task_anc.shape[2], task_anc.shape[3]]
            for i in range(args.datanum):
                distribution = torch.distributions.normal.Normal(torch.nn.functional.relu(net.mu.detach()[i, 0])+1e-6,
                                                                 torch.nn.functional.relu(net.sigma.detach()[i, 0])+1e-6)
                sample = distribution.sample(sample_shape = sample_shape).permute(0, 3, 1, 2)
                sampled_x.append(sample)
                sampled_y.append(torch.LongTensor([i]*sample.shape[0]))
            sampled_x = torch.cat(sampled_x, dim=0).to(args.device)
            sampled_y = torch.cat(sampled_y, dim=0).to(args.device)
            sampled_output = net.classifier(sampled_x.reshape(sampled_x.shape[0], -1))
            sampledloss = loss(sampled_output, sampled_y)
            sampleacc = (torch.argmax(sampled_output, dim=1) == sampled_y).float().mean()
        else:
            sampledloss = torch.zeros(1).to(args.device)
            sampleacc = torch.zeros(1)

        miniloss = (clsloss * hyper[0] +
                    distloss * hyper[1] +
                    subjectloss * hyper[2] +
                    sampledloss * hyper[3])

        if train == 'train' or train == 'fine' or train == 'RS':
            optimizer.zero_grad()
            miniloss.backward()
            optimizer.step()

            if args.cdist != 0 and hyper[1] != 0:
                for i in range(args.datanum):
                    if net.mu.shape[1] == 1:
                        net.mu[i].data -= dist_mu[i] * args.cdist
                        net.sigma[i].data -= dist_sigma[i] * args.cdist

        output = torch.nn.functional.softmax(output, 1)
        answer = torch.argmax(output, 1) == y
        miniacc = answer.float().mean()
        acc_epoch += miniacc.item() * output.shape[0]
        loss_epoch += miniloss.item() * output.shape[0]
        count_epoch += output.shape[0]

        print(f'Acc: {round(miniacc.item(), 5)} - '
              f'CLS: {round(clsloss.item(), 5)} - '
              f'Dist: {round(distloss.item(), 5)} - '
              f'Subject: {round(subjectloss.item(), 5)} - '
              f'Sample: {round(sampledloss.item(), 5)} - '
              f'Sacc: {round(sampleacc.item(), 5)}', end='\r')

        loss_print['Total'] += miniloss.item() * output.shape[0]
        loss_print['CLS'] += clsloss.item() * output.shape[0]
        loss_print['Dist'] += distloss.item() * output.shape[0]
        loss_print['Sub'] += subjectloss.item() * output.shape[0]
        loss_print['Sample'] += sampledloss.item() * output.shape[0]

    for k, v in loss_print.items():
        loss_print[k] /= count_epoch

    return x, y, output, acc_epoch/count_epoch, loss_print, count_epoch


def generate_signal_from_RS(net, data_loader, loss, hyper, args, train='train'):
    net.eval()
    dataset_RS = []
    dataset_RS_update = []
    dataset_RS_label = []
    for i_iter, sample_train in enumerate(data_loader):
        RS_init_list = []
        RS_update = []
        RS_update_label = []

        for label in range(args.datanum):
            RS = sample_train.to(args.device)
            output_init, taskf_init, subf_init = net.forward(RS, train)
            subf_init = subf_init.detach()
            if subf_init.dim() != 2:
                subf_init = subf_init.reshape(subf_init.shape[0], -1)

            proto = net.mu.data[:, 0]
            RS_init = torch.rand_like(RS, requires_grad=True, dtype=RS.dtype, device=args.device)
            optimizer = torch.optim.Adam([RS_init], lr=0.005)
            pdist = torch.nn.PairwiseDistance()
            u_iter = 0
            pseudo_label = torch.zeros(args.datanum)
            pseudo_label[label] = 1
            run = 0
            while run == 0:
                optimizer.zero_grad()
                output, feature, subf= net.forward(RS_init, train)
                clsloss = loss(output, torch.tensor([label]).to(args.device))

                ## proto
                metrics = []
                dist_ = torch.cdist(feature.mean(dim=(2, 3)), proto.detach())[0]
                for j in range(1, args.datanum):
                    metric_mu = torch.max(torch.zeros(1).to(args.device), dist_[label] - dist_[label - j] + 1)
                    metric = metric_mu
                    metrics.append(metric)

                f_cls = dist_[label].mean() + torch.cat(metrics, dim=0).mean()
                if subf.dim() != 2:
                    subf = subf.reshape(subf.shape[0], -1)
                f_sub = pdist(subf, subf_init)
                miniloss = f_sub * 10 + clsloss + f_cls

                miniloss.backward()
                optimizer.step()

                if u_iter == 300:
                    RS_init_list.append(RS.detach().cpu())
                    RS_update.append(RS_init.data.detach().cpu())
                    RS_update_label.append(pseudo_label)
                    run = 1
                else:
                    u_iter += 1

        RS_init_ = torch.cat(RS_init_list, dim=0)
        RS_update_ = torch.cat(RS_update, dim=0)
        RS_update_label_ = torch.stack(RS_update_label, dim=0)
        dataset_RS.append(RS_init_)
        dataset_RS_update.append(RS_update_)
        dataset_RS_label.append(RS_update_label_)

    return dataset_RS, dataset_RS_update, dataset_RS_label