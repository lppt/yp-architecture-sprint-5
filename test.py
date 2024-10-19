import numpy as np

# Параметры Q-Learning
alpha = 0.1  # Скорость обучения
gamma = 0.9  # Коэффициент дисконтирования
epsilon = 0.1  # Параметр для ε-greedy стратегии (исследование)

# Определение среды (простой лабиринт 4x4)
grid_size = 4
reward_matrix = np.zeros((grid_size, grid_size))
reward_matrix[grid_size-1, grid_size-1] = 1  # Целевая точка с вознаграждением

# Инициализация Q-таблицы (состояния x действия)
q_table = np.zeros((grid_size * grid_size, 4))  # 4 действия: вверх, вниз, влево, вправо

# Вспомогательные функции для конвертации состояния и действий
def state_to_position(state):
    return state // grid_size, state % grid_size

def position_to_state(position):
    return position[0] * grid_size + position[1]

def is_terminal_state(state):
    return reward_matrix[state_to_position(state)] == 1

def get_available_actions(state):
    actions = []
    row, col = state_to_position(state)
    if row > 0: actions.append(0)  # вверх
    if row < grid_size - 1: actions.append(1)  # вниз
    if col > 0: actions.append(2)  # влево
    if col < grid_size - 1: actions.append(3)  # вправо
    return actions

def take_action(state, action):
    row, col = state_to_position(state)
    if action == 0 and row > 0: row -= 1  # вверх
    elif action == 1 and row < grid_size - 1: row += 1  # вниз
    elif action == 2 and col > 0: col -= 1  # влево
    elif action == 3 and col < grid_size - 1: col += 1  # вправо
    new_state = position_to_state((row, col))
    return new_state, reward_matrix[(row, col)]

# Цикл Q-Learning
episodes = 1500
for episode in range(episodes):
    state = np.random.randint(0, grid_size * grid_size)  # Случайное начальное состояние
    while not is_terminal_state(state):
        # Выбор действия (ε-greedy стратегия)
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(get_available_actions(state))  # Исследование
        else:
            action = np.argmax(q_table[state, :])  # Эксплуатация

        # Выполнение действия
        new_state, reward = take_action(state, action)

        # Обновление Q-значения
        best_future_q = np.max(q_table[new_state, :])
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * best_future_q - q_table[state, action])

        state = new_state  # Переход к новому состоянию

# Вывод итоговой Q-таблицы
import pandas as pd
q_table_df = pd.DataFrame(q_table, columns=["Вверх", "Вниз", "Влево", "Вправо"])
print(q_table_df)