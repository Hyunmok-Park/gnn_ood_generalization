import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import progressbar
import time
class GRU(nn.Module):
    def __init__(self, hidden_dim):
        super(GRU, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.h_hat = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.Tanh()
        )

    def forward(self, h, m_t_0, m_t_1):
        a = torch.cat((h, m_t_0, m_t_1), 2)  # (N,3D)
        z = self.update_gate(a)  # (N, D)
        r = self.reset_gate(a)  # (N, D)
        joined = torch.cat((r * h, m_t_0, m_t_1), 2)  # (N, 3D)
        h_hat = self.h_hat(joined)  # (N, D)

        updated_h = (1 - z) * h + z * h_hat  # (N, D)
        return updated_h


class GNN(nn.Module):
    def __init__(self, Graph):
        super(GNN, self).__init__()
        self.hidden_dim = 5
        self.n_node = len(Graph.nodes())
        self.message_step = 10

        self.message = Message(self.hidden_dim)

        self.gru = GRU(self.hidden_dim)

        self.h_update = nn.Sequential(
            nn.Linear(3 * self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, J, b):
        self.batch_size = len(J)
        J_ = []
        edge_list = []
        for _ in J:
            _J = []
            edge_list_ = []
            for i, x in enumerate(_):
                for j, y in enumerate(x):
                    if (y != 0):
                        edge_list_.append((i, j))
                        _J.append(float(y))
            J_.append(_J)
            edge_list.append(edge_list_)
        J_msg = torch.tensor(J_).view(self.batch_size, -1, 1).double()
        index_in, index_out = torch.tensor(edge_list)[0][:, 0], torch.tensor(edge_list)[0][:, 1]
        b_in = b[:, index_in]
        b_out = b[:, index_out]

        self.h_t_0 = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()
        self.h_t_1 = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()
        self.h = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()
        self._ = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()

        message_t_0 = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()
        message_t_1 = torch.zeros(self.batch_size, self.n_node, self.hidden_dim).double()

        for i in range(self.message_step):
            me_t_0 = self.message(self.h_t_0[:, index_out], J_msg, b_in, b_out)  # (E, D)
            me_t_1 = self.message(self.h_t_1[:, index_out], J_msg, b_in, b_out)  # (E, D)

            _ = index_out.view(-1, 1).expand(-1, self.hidden_dim)
            scatter_index = index_out.view(-1, 1).expand(-1, self.hidden_dim)
            for i in range(self.batch_size - 1):
                scatter_index = torch.cat((scatter_index, _), 0)
            scatter_index = scatter_index.view(self.batch_size, -1, self.hidden_dim)

            message_t_0 = message_t_0.scatter_add(1, scatter_index, me_t_0)  # (N,D)
            message_t_1 = message_t_0.scatter_add(1, scatter_index, me_t_1)  # (N,D)

            # self.h = self.h_update(torch.cat((self.h_t_1, message_t_0, message_t_1), 2)) #(N, D + D + D)
            self.h = self.gru(self.h_t_1, message_t_0, message_t_1)

            self._ = self.h_t_1
            self.h_t_1 = self.h
            self.h_t_0 = self._

        output = self.out(torch.cat((self.h, b), 2))
        output = output.view(self.batch_size, 1, -1)
        return output


class Message(nn.Module):
    def __init__(self, hidden_dim):
        super(Message, self).__init__()
        self.hidden_dim = hidden_dim

        self.message_nn = nn.Sequential(
            nn.Linear(hidden_dim + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Linear(64,64),
            # nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, h, J, b_in, b_out):
        _ = torch.cat((h, b_in, b_out, J, -J), 2)  # (D+1+1)
        message = self.message_nn(_)
        return message


criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss(reduction='batchmean')
criterion3 = nn.MSELoss()


def myloss(target, output):
    P = target.transpose(1, 2)
    Q = output.transpose(1, 2)
    P = torch.cat((P, 1 - P), 2)
    Q = torch.cat((Q, 1 - Q), 2)
    loss = nn.KLDivLoss(reduction='batchmean')(Q.log(), P)
    return loss


class EarlyStop:
    def __init__(self):
        self.Best_score = None
        self.stop = False
        self.count = 0

    def __call__(self, loss):
        if self.Best_score is None:
            self.Best_score = loss
            self.stop = False
            return self.stop
        elif (loss < self.Best_score):
            self.Best_score = loss
            if (self.Best_score < 0.006):
                self.count = self.count + 1
                if (self.count == 20):
                    self.stop = True
                    return self.stop
            return self.stop


def train(net, optimizer, J, b, target):
    LOSS = []
    bar = progressbar.ProgressBar(widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') ', ])
    net.train()
    # stopper = EarlyStop()
    for i in bar(range(150)):
        output = net(J, b)
        loss = myloss(target, output)
        #loss = criterion3(target, output)
        #loss = criterion2(target, output.log())
        l = loss.detach()
        LOSS.append(float(l))
        '''
        stop = stopper(loss)
        if(stop):
          print("\nEarly stop")
          break
        '''
        net.zero_grad()
        loss.backward()
        optimizer.step()
        time.sleep(0.001)

    plt.figure(0)
    plt.scatter([i for i in range(len(LOSS))], LOSS)
    print("\nLoss:", float(loss))