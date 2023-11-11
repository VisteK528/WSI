def ocen_zysk_dla_ruchu(plansza, gracz, kolumna):
    zysk = 0

    # Sprawdź dostępne ruchy w danej kolumnie
    for row in range(5, -1, -1):
        if plansza[row][kolumna] == 0:
            plansza[row][kolumna] = gracz
            break

    # Sprawdź poziome linie
    for row in range(6):
        for col in range(4):
            segment = plansza[row][col:col + 4]
            if segment.count(gracz) == 4:
                zysk += 1000
            elif segment.count(gracz) == 3 and segment.count(0) == 1:
                zysk += 5
            elif segment.count(gracz) == 2 and segment.count(0) == 2:
                zysk += 2

    # Sprawdź pionowe linie
    for col in range(7):
        for row in range(3):
            segment = [plansza[row + i][col] for i in range(4)]
            if segment.count(gracz) == 4:
                zysk += 1000
            elif segment.count(gracz) == 3 and segment.count(0) == 1:
                zysk += 5
            elif segment.count(gracz) == 2 and segment.count(0) == 2:
                zysk += 2

    # Sprawdź ukosne linie (pierwsza przekątna)
    for row in range(3):
        for col in range(4):
            segment = [plansza[row + i][col + i] for i in range(4)]
            if segment.count(gracz) == 4:
                zysk += 1000
            elif segment.count(gracz) == 3 and segment.count(0) == 1:
                zysk += 5
            elif segment.count(gracz) == 2 and segment.count(0) == 2:
                zysk += 2

    # Sprawdź ukosne linie (druga przekątna)
    for row in range(3):
        for col in range(3, 7):
            segment = [plansza[row + i][col - i] for i in range(4)]
            if segment.count(gracz) == 4:
                zysk += 1000
            elif segment.count(gracz) == 3 and segment.count(0) == 1:
                zysk += 5
            elif segment.count(gracz) == 2 and segment.count(0) == 2:
                zysk += 2

    return zysk

def evaluate_connected(connected: int) -> float:
    two_in_line = 2
    three_in_line = 5
    winning_condition = 1000
    opp_two_in_line = -2
    opp_winning_condition = -100

    if connected == 2:
        return two_in_line
    elif connected == 3:
        return three_in_line
    elif connected == 4:
        return winning_condition
    elif connected == -2:
        return opp_two_in_line
    elif connected == -3:
        return opp_winning_condition
    else:
        return 0

def find_connected(board, column, player):
    row = 0
    value = 0

    for i in range(len(board)):
        if board[i][column] != 0:
            break
        else:
            row = i

    vertically_connected = 1
    for i in range(row + 1, min(len(board), row + 4)):
        if board[i][column] == player:
            vertically_connected += 1
        else:
            break

    board[row][column] = player
    horizontally_connected_possibilities = [0 for _ in range(min(column, len(board) - column) + 1)]
    for col in range(min(column, len(board) - column) + 1):
        segment = board[row][max(0, column - 3) + col:min(max(0, column - 3) + col + 4, 7)]
        if segment.count(gracz) == 4:
            horizontally_connected_possibilities[col] = 4
        elif segment.count(gracz) == 3 and segment.count(0) == 1:
            horizontally_connected_possibilities[col] = 3
        elif segment.count(gracz) == 2 and segment.count(0) == 2:
            horizontally_connected_possibilities[col] = 2
    board[row][column] = 0

    # Check diagonally (left) down for positive
    diagonally_down_left_connected = 1
    diagonal_moves = min(min(len(board) - 1, row + 4) - row, column)
    for i in range(diagonal_moves + 1):
        if board[row + i][column - i] == player:
            diagonally_down_left_connected += 1
        elif board[row + i][column - i] != 0:
            diagonally_down_left_connected = 1

    # Check diagonally (right) down for positive
    diagonally_down_right_connected = 1
    diagonal_moves = min(min(len(board) - 1, row + 4) - row, len(board[0]) - 1 - column)
    for i in range(diagonal_moves + 1):
        if board[row + i][column + i] == player:
            diagonally_down_right_connected += 1
        elif board[row + i][column + i] != 0:
            diagonally_down_right_connected = 1

    value += evaluate_connected(diagonally_down_right_connected)

    # Check diagonally (left) up for positive
    diagonally_up_left_connected = 1
    diagonal_moves = min(row - max(0, row - 4), column)

    for i in range(1, diagonal_moves + 1):
        if board[row - i][column - i] == player:
            diagonally_up_left_connected += 1
        elif board[row - i][column - i] != 0:
            diagonally_up_left_connected = 1

    value += evaluate_connected(diagonally_up_left_connected)

    # Check diagonally (right) up for positive
    diagonally_up_right_connected = 1
    diagonal_moves = min(row - max(0, row - 4), len(board[0]) - 1 - column)

    for i in range(1, diagonal_moves + 1):
        if board[row - i][column + i] == player:
            diagonally_up_right_connected += 1
        elif board[row - i][column + i] != 0:
            diagonally_up_right_connected = 1

    value += evaluate_connected(diagonally_up_right_connected)



def my_evaluate_move_2(board, player, column) -> float | None:
    if board[0][column] != 0:
        return None

    center = 4
    two_in_line = 2
    three_in_line = 5
    winning_condition = 1000
    opp_two_in_line = -2
    opp_winning_condition = -100

    value = 0
    row = 0

    for i in range(len(board)):
        if board[i][column] != 0:
            break
        else:
            row = i

    # Center bonus
    if column == 3:
        value += center
    # Check vertical for positive
    vertically_connected = 1
    for i in range(row+1, min(len(board), row+4)):
        if board[i][column] == player:
            vertically_connected += 1
        else:
            break

    value += evaluate_connected(vertically_connected)

    #print("Range: ", min(column, len(board) - column)+1, max(0, column-3))
    board[row][column] = player
    for col in range(min(column, len(board) - column)+1):
        segment = board[row][max(0, column-3)+col:min(max(0, column-3)+col + 4, 7)]
        if segment.count(gracz) == 4:
            value += 1000
        elif segment.count(gracz) == 3 and segment.count(0) == 1:
            value += 5
        elif segment.count(gracz) == 2 and segment.count(0) == 2:
            value += 2
            print(segment)
    board[row][column] = 0

    # Check diagonally (left) down for positive
    diagonally_down_left_connected = 1
    diagonal_moves = min(min(len(board)-1, row + 4)-row, column)
    for i in range(diagonal_moves+1):
        if board[row+i][column-i] == player:
            diagonally_down_left_connected += 1
        elif board[row+i][column-i] != 0:
            diagonally_down_left_connected = 1

    value += evaluate_connected(diagonally_down_left_connected)

    # Check diagonally (right) down for positive
    diagonally_down_right_connected = 1
    diagonal_moves = min(min(len(board) - 1, row + 4) - row, len(board[0])-1-column)
    for i in range(diagonal_moves + 1):
        if board[row + i][column + i] == player:
            diagonally_down_right_connected += 1
        elif board[row + i][column + i] != 0:
            diagonally_down_right_connected = 1

    value += evaluate_connected(diagonally_down_right_connected)

    # Check diagonally (left) up for positive
    diagonally_up_left_connected = 1
    diagonal_moves = min(row - max(0, row - 4), column)

    for i in range(1, diagonal_moves + 1):
        if board[row - i][column - i] == player:
            diagonally_up_left_connected += 1
        elif board[row - i][column - i] != 0:
            diagonally_up_left_connected = 1

    value += evaluate_connected(diagonally_up_left_connected)

    # Check diagonally (right) up for positive
    diagonally_up_right_connected = 1
    diagonal_moves = min(row - max(0, row - 4), len(board[0])-1-column)

    for i in range(1, diagonal_moves + 1):
        if board[row - i][column + i] == player:
            diagonally_up_right_connected += 1
        elif board[row - i][column + i] != 0:
            diagonally_up_right_connected = 1

    value += evaluate_connected(diagonally_up_right_connected)

    # Sprawdź ukosne linie (pierwsza przekątna)
    for row in range(3):
        for col in range(4):
            segment = [plansza[row + i][col + i] for i in range(4)]
            if segment.count(gracz) == 4:
                value += 1000
            elif segment.count(gracz) == 3 and segment.count(0) == 1:
                value += 5
            elif segment.count(gracz) == 2 and segment.count(0) == 2:
                value += 2

    return value


# Przykładowa plansza (0 - puste, 1 - gracz 1, 2 - gracz 2)
plansza = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0],
    [0, 0, 2, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [2, 1, 2, 1, 0, 0, 0]
]

gracz = 1  # Gracz, dla którego chcemy ocenić zysk
kolumna = 3  # Kolumna, w której chcemy wykonać ruch

#zysk = ocen_zysk_dla_ruchu(plansza, gracz, kolumna)
#my_evaluate_move(plansza, gracz, 3)
values = [my_evaluate_move_2(plansza, 2, x) for x in range(7)]
print("Evaluations: ", values)

