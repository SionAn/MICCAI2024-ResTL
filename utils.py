import os
import shutil
import sys
import torch

class find_best_model():
    def __init__(self, path, args):
        self.total_best_acc = 0
        self.total_best_loss = 100000000
        self.total_best_epoch = 0
        if not os.path.exists(path):
            os.makedirs(path)
        self.f = open(os.path.join(path, 'log.txt'), 'w')
        self.args = args
        self.f.write('******Info*****\n')
        for arg, value in sorted(vars(self.args).items()):
            self.f.write('%s, %s\n' %(arg, str(value)))
        self.f.write('***************\n')

    def update(self, model, input_path, epoch, acc, loss):
        if self.total_best_loss >= loss:
            self.total_best_loss = loss
            self.total_best_epoch = epoch
            self.total_best_acc = acc
            self.model_save(model, input_path, 'model-best.pth')

        self.f.write('Epoch: %d, Acc: %f, Loss: %f\n'
                     %(epoch+1, round(acc, 7), round(loss, 7)))

    def model_save(self, model, input_path, name):
        torch.save(model.state_dict(), os.path.join(input_path, name))

    def code_copy(self, path):
        file_list = os.listdir('.')
        for i in range(len(file_list)):
            if file_list[i][-3:] == '.py':
                self.file_copy('.', path, file_list[i])

    def file_copy(self, input_path, output_path, file_name):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        shutil.copyfile(input_path + '/' + file_name, output_path + '/' + file_name)

    def training_finish(self, model, input_path):
        self.model_save(model, input_path, 'model-last.pth')
        self.f.write('Best epoch\n')
        self.f.write('Epoch: %d, Acc: %f, Loss: %f\n'
                     % (self.total_best_epoch+1, round(self.total_best_acc, 7),
                        round(self.total_best_loss, 7)))
        self.f.close()

    def earlystop(self, model, path, epoch, num):
        if self.total_best_epoch+num == epoch:
            self.training_finish(model, path)
            print("Training stop")
            sys.exit(1)

class test_model():
    def __init__(self, path, args, filename='result'):
        os.makedirs(path, exist_ok=True)
        self.f = open(os.path.join(path, filename+'.txt'), 'w')
        self.args = args
        self.f.write('******Info*****\n')
        for arg, value in sorted(vars(self.args).items()):
            self.f.write('%s, %s\n' %(arg, str(value)))
        self.f.write('***************\n')

    def write_result(self, idx, epoch, acc, loss):
        self.f.write('idx: %d/%d, Acc: %f, Loss: %f\n' %(idx, epoch, acc, loss))

    def total_result(self, key, idx, acc, loss):
        self.f.write('%s, Test: %d, Acc: %f, Loss: %f\n' % (key, idx, acc, loss))
