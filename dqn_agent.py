import torch
from stock import get_historical_prices, get_historical_vix
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt


class DQN(nn.Module):
    """
      A neural network for estimating value of the Q function.
      """

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fully_connected_1 = nn.Linear(state_size, 256)
        self.fully_connected_2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.fully_connected_1(x))
        return self.fully_connected_2(x)


class RLAgent:
    """
      The Stock Trading Agent, which would be responsible for making decisions
    """

    def __init__(self, state_size, window_size, closing_prices, volumes, vixes, skip, batch_size):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.closing_prices = closing_prices
        self.vixes = vixes
        self.skip = skip
        self.action_size = 3  # Action 0 - hold, Action 1 - Buy, Action 2 - Sell
        self.batch_size = batch_size
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.volumes = volumes

        # Discount factor, time value of money,
        # current profit will be always useful that in any future profits
        self.gamma = 0.7

        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, self.action_size).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()

    def take_action(self, state):
        """Take a random variable between 0 and 1, if value is less that 0.5, do the exploration 
        (select a random action from buy / sell or hold), if not do the exploitation (find the action 
         which gives maximum Q value for the state) """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def get_state(self, t):
        """
        State now includes:
        - Price movements for the last few days (window size)
        - Volume
        - VIX
        """
        window_size = self.window_size + 1
        starting_index = t - window_size + 1
        if starting_index >= 0:
            block = self.closing_prices[starting_index:t + 1]
            volumes = self.volumes[starting_index:t + 1]
            vixes = self.vixes[starting_index:t + 1]

        else:
            block = -starting_index * \
                [self.closing_prices[0]] + self.closing_prices[0:t + 1]
            volumes = -starting_index * \
                [self.volumes[0]] + self.volumes[0:t + 1]
            vixes = -starting_index * [self.vixes[0]] + self.vixes[0:t + 1]

        price_movements = [block[i + 1] - block[i]
                           for i in range(window_size - 1)]

        # Construct the state as a numpy array
        state = price_movements
        #state.extend(volumes)
        #state.extend(vixes)
        return np.array([state])

    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        for t in range(0, len(self.closing_prices) - 1, self.skip):
            action = self.take_action(state)
            next_state = self.get_state(t + 1)

            if action == 1 and initial_money >= self.closing_prices[t] and t < (len(self.closing_prices) - self.half_window):
                inventory.append(self.closing_prices[t])
                initial_money -= self.closing_prices[t]
                states_buy.append(t)
                print('Day %d: Buy 1 unit at price %f, Total Balance %f' %
                      (t, self.closing_prices[t], initial_money))

            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.closing_prices[t]
                states_sell.append(t)
                try:
                    invest = (
                        (self.closing_prices[t] - bought_price) / bought_price) * 100
                except ZeroDivisionError:
                    invest = 0
                print(
                    'Day %d, sell 1 unit at price %f, Investment %f %%, Total Balance %f'
                    % (t, self.closing_prices[t], invest, initial_money)
                )

            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        replay_size = len(mini_batch)
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])

        states = torch.FloatTensor(states).to(self.device)
        new_states = torch.FloatTensor(new_states).to(self.device)

        Q = self.model(states)
        Q_new = self.model(new_states)

        X = torch.zeros((replay_size, self.state_size), device=self.device)
        Y = torch.zeros((replay_size, self.action_size), device=self.device)

        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i].clone()
            target[action] = reward
            if not done:
                target[action] += self.gamma * torch.max(Q_new[i]).item()
            X[i] = torch.FloatTensor(state).to(self.device)
            Y[i] = target

        self.optimizer.zero_grad()
        cost = self.criterion(Q, Y)
        cost.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return cost.item()

    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            for t in range(0, len(self.closing_prices) - 1, self.skip):
                action = self.take_action(state)
                next_state = self.get_state(t + 1)

                if action == 1 and starting_money >= self.closing_prices[t] and t < (len(self.closing_prices) - self.half_window):
                    inventory.append(self.closing_prices[t])
                    starting_money -= self.closing_prices[t]

                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.closing_prices[t] - bought_price
                    starting_money += self.closing_prices[t]

                invest = ((starting_money - initial_money) / initial_money)
                self.memory.append((state, action, invest,
                                    next_state, starting_money < initial_money))
                state = next_state
                batch_size = min(self.batch_size, len(self.memory))
                cost = self.replay(batch_size)
            if (i + 1) % checkpoint == 0:
                print('epoch: %d, Total rewards: %f, Cost: %f, Total Money: %f' % (
                    i + 1, total_profit, cost, starting_money))


def main():
    stock_prices = get_historical_prices('AAPL', '2024-01-01', '2024-08-30')
    vix_data = get_historical_vix('2024-01-01', '2024-08-30')

    initial_money = 10000
    closing_prices = stock_prices.Close.values.tolist()
    volumes = stock_prices.Volume.values.tolist()
    vixes = vix_data.Close.values.tolist()

    window_size = 30
    skip = 1
    batch_size = 32
    state_size = 3 * (window_size - 1)
    agent = RLAgent(state_size=state_size, window_size=window_size, closing_prices=closing_prices,
                    volumes=volumes, vixes=vixes, skip=skip, batch_size=batch_size)
    agent.train(iterations=200, checkpoint=10, initial_money=initial_money)

    states_buy, states_sell, total_gains, invest = agent.buy(
        initial_money=initial_money)

    # starting backtesting to see how have we performed
    fig = plt.figure(figsize=(15, 5))
    plt.plot(closing_prices, color='r', lw=2.)
    plt.plot(closing_prices, '^', markersize=10, color='m',
             label='buying signal', markevery=states_buy)
    plt.plot(closing_prices, 'v', markersize=10, color='k',
             label='selling signal', markevery=states_sell)
    plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
    plt.legend()
    plt.savefig('./backtesting_plot.jpeg')

    pass


if __name__ == "__main__":
    main()
