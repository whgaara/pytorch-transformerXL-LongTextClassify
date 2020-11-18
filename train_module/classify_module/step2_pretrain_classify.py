import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformerXL.data.classify_dataset import *
from transformerXL.layers.TransformerXL import TransformerXL


if __name__ == '__main__':
    onehot_type = False
    labelcount = int(open(Assistant, 'r', encoding='utf-8').read().strip().split(',')[1])
    transformerXL = TransformerXL(kinds_num=labelcount).to(device)

    dataset = TransformerXLDataSet(CorpusPath)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=True)
    testset = TransformerXLTestSet(EvalPath)

    optim = Adam(transformerXL.parameters(), lr=LearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(Epochs):
        # train
        transformerXL.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['desc_segments']
            segment_ids = data['type_segments']
            label = data['token_ids_labels']
            mlm_output = transformerXL(input_token, segment_ids)
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()
        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = PretrainPath + '.ep%d' % epoch
        torch.save(transformerXL.cpu(), output_path)
        transformerXL.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            transformerXL.eval()
            test_count = 0
            accuracy_count = 0
            for test_data in testset:
                test_count += 1
                input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                input_token_list = input_token.tolist()
                input_len = len([x for x in input_token_list[0] if x]) - 2
                label = test_data['token_ids_labels'].tolist()
                mlm_output = transformerXL(input_token, segment_ids)
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 1).indices.squeeze(0).tolist()

                # 累计数值
                if label == output_topk[0]:
                    accuracy_count += 1

            if test_count:
                acc_rate = float(accuracy_count) / float(test_count)
                acc_rate = round(acc_rate, 2)
                print('判断正确率：%s' % acc_rate)
