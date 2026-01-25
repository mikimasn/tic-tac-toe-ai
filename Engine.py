from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from torch.distributions.dirichlet import Dirichlet

gameStateType = TypeVar("gameStateType")
gameMoveType = TypeVar("gameMoveType")


@dataclass
class MCTSNode(Generic[gameStateType, gameMoveType]):
    visitcount: int
    eval: float
    gameState: gameStateType
    possibleMoves: list[gameMoveType]
    possibleMovesHash: list[int]
    possibleMovesPropabilityDist: list[float]
    possibleMovesVisitCount: list[int]
    possibleMovesEvalSum: list[float]
    depth: int


class MCTS(ABC, Generic[gameStateType, gameMoveType]):
    def __init__(self, start_state: gameStateType, history_size: int, cpuct: float = 1, training_mode: bool = False, epsilon: float = 0, training_data_temp: float = 1.0):
        self.transpositionTable: dict[int, MCTSNode[gameStateType, gameMoveType]] = {}
        self.currenthistory: list[gameStateType] = []
        self.historysize = history_size
        self.rootnode: int = self._add_node(start_state, None, 0, [])
        self.cpuct: float = cpuct
        self.training_mode: bool = training_mode
        self.training_data: list[tuple[gameStateType, torch.Tensor]] = []
        self.currenthistory.append(self.transpositionTable[self.rootnode].gameState)
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon
        self.first = True
        assert training_data_temp > 0
        self.training_data_temp = training_data_temp

    def step(self, moveidx: int):
        rootnode = self.transpositionTable[self.rootnode]
        if self.training_mode:
            props = torch.Tensor(rootnode.possibleMovesVisitCount)
            props = props ** (1 / self.training_data_temp)
            props = props / props.sum()
            self.training_data.append((rootnode.gameState, props))
        if not rootnode.possibleMovesHash[moveidx] == -1:
            self.rootnode = rootnode.possibleMovesHash[moveidx]
        else:
            move = rootnode.possibleMoves[moveidx]
            gameState = rootnode.gameState
            self.rootnode = self._add_node(gameState, move, rootnode.depth + 1, [])
        self.prune(rootnode)
        rootnode = self.transpositionTable[self.rootnode]
        self.currenthistory.append(rootnode.gameState)
        self.first = False

    def select(self, temp: float = 1) -> int:
        currentNode = self.transpositionTable[self.rootnode]
        if temp == 0:
            return int(torch.argmax(torch.Tensor(currentNode.possibleMovesVisitCount)).item())
        propvec = torch.Tensor(currentNode.possibleMovesVisitCount) ** (1/ temp)
        propvec = propvec / propvec.sum()
        if self.first and self.epsilon > 0:
            noise = Dirichlet(torch.full((len(currentNode.possibleMovesVisitCount),),0.2)).sample()
            propvec = (1-self.epsilon) * propvec + self.epsilon * noise
        return int(torch.multinomial(propvec, 1).item())
    @abstractmethod
    def prune(self, rootNode: MCTSNode[gameStateType, gameMoveType]):
        pass

    def change_cpuct(self, cpuct: float):
        self.cpuct = cpuct

    def perform_iteration(self):
        actNodeHash, actNode = self.rootnode, self.transpositionTable[self.rootnode]
        path: list[tuple[int, int]] = []
        while not actNodeHash == -1:
            actNodeVisitCount = (actNode.visitcount) ** 0.5
            bestidx, bestvalue = 0, -1e9
            if len(actNode.possibleMoves) == 0:
                self._backprop(path, actNode.eval)
                break
            cpuct = self.cpuct
            for idx, hash in enumerate(actNode.possibleMovesHash):
                value = 0
                if hash == -1:
                    value = cpuct * actNode.possibleMovesPropabilityDist[idx] * actNodeVisitCount
                else:
                    value = actNode.possibleMovesEvalSum[idx] / actNode.possibleMovesVisitCount[idx] + cpuct * \
                            actNode.possibleMovesPropabilityDist[idx] * actNodeVisitCount / (
                                    1 + actNode.possibleMovesVisitCount[idx])
                if value > bestvalue:
                    bestvalue = value
                    bestidx = idx
            path.append((actNodeHash, bestidx))
            if actNode.possibleMovesHash[bestidx] == -1:
                actNode.possibleMovesHash[bestidx] = self._add_node(actNode.gameState, actNode.possibleMoves[bestidx],
                                                                    actNode.depth + 1, path)
                break
            actNodeHash = actNode.possibleMovesHash[bestidx]
            actNode = self.transpositionTable[actNodeHash]

    def _add_node(self, gamestate: gameStateType, move: gameMoveType | None, depth: int,
                  path_from_root: list[tuple[int, int]]) -> int:
        if move is not None:
            newPositon = self._apply_move(gamestate, move)
        else:
            newPositon = gamestate
        newPositionHash = self._hash_position(newPositon)
        if self.transpositionTable.get(newPositionHash) is not None:
            self.evaluate(newPositionHash, path_from_root)
            return newPositionHash
        newNode = MCTSNode(1, 0, newPositon, self.get_possible_moves(newPositon), [], [], [], [], depth)
        newNode.possibleMovesHash, newNode.possibleMovesVisitCount, newNode.possibleMovesEvalSum = [-1 for _ in
                                                                                                    newNode.possibleMoves], [
            0 for _ in newNode.possibleMoves], [0 for _ in newNode.possibleMoves]
        self.transpositionTable[newPositionHash] = newNode
        self.evaluate(newPositionHash, path_from_root)
        return newPositionHash

    def _backprop(self, path_from_root: list[tuple[int, int]], value: float):
        value = -value #value represents eval at the leaf node which SHOULDN'T be inside path_from_root due to that we need to invert value to represent the perspective of the lowest node in the path
        for node, idx in reversed(path_from_root):
            mcNode = self.transpositionTable[node]
            mcNode.possibleMovesVisitCount[idx] += 1
            mcNode.possibleMovesEvalSum[idx] += value
            mcNode.visitcount += 1
            mcNode.eval += value
            value = -value

    @abstractmethod
    def get_possible_moves(self, gameState: gameStateType) -> list[gameMoveType]:
        pass

    @abstractmethod
    def _apply_move(self, gameState: gameStateType, move: gameMoveType) -> gameStateType:
        pass

    @abstractmethod
    def _hash_position(self, gamestate: gameStateType) -> int:
        pass

    # YOU MUST BACKPROP AFTER EVAL
    @abstractmethod
    def evaluate(self, node: int, path_from_root: list[tuple[int, int]]):
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        pass

    @abstractmethod
    def game_result(self) -> int:
        pass

    @abstractmethod
    def get_training_data(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass


@dataclass
class GameState():
    crossNowPlaying: bool
    nowallowed: int
    board: np.ndarray
    wins: np.ndarray
    finished: bool

    def embed_board(self) -> torch.Tensor:
        Cpos: list[list[float]]
        Opos: list[list[float]]
        Cpos, Opos = [[0 for _ in range(9)] for _ in range(9)], [[0 for _ in range(9)] for _ in range(9)]
        currenttoplay = 1 if self.crossNowPlaying else 2
        for idx, ele in enumerate(self.board):
            if ele == currenttoplay:
                Cpos[idx // 9][idx % 9] = 1
            elif not ele == 0:
                Opos[idx // 9][idx % 9] = 1
        moveMask: list[list[float]] = [[0 if not self.nowallowed < 0 else 1 for _ in range(9)] for _ in range(9)]
        if self.nowallowed >= 0:
            rx, ry = self.nowallowed % 3, self.nowallowed // 3
            rx, ry = rx * 3, ry * 3
            for x in range(3):
                for y in range(3):
                    moveMask[ry + y][rx + x] = 1
        for x in range(9):
            for y in range(9):
                if Cpos[x][y] != 0 or Opos[x][y] != 0:
                    moveMask[x][y] = 0
        return torch.Tensor([Cpos, Opos, moveMask])

    def visualize_board(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for y in range(9):
            row = ""
            for x in range(9):
                idx = y*9 + x
                row += symbols[self.board[idx]] + " "
                if x % 3 == 2 and x != 8:
                    row += "| "
            print(row)
            if y % 3 == 2 and y != 8:
                print("- " * 11)


class MCTSForestParticipant(MCTS[gameStateType, gameMoveType]):
    @abstractmethod
    def postevaluate(self, input: torch.Tensor, prop: torch.Tensor, v: torch.Tensor, id: int):
        pass

    def set_forest_identifier(self, identifier: int):
        self._forest_identifier = identifier


class MCTSForest(Generic[gameStateType, gameMoveType]):
    def __init__(self, model: nn.Module, device: torch.device, shards:int = 1):
        self.model, self.device = model, device
        self.games: list[MCTSForestParticipant[gameStateType, gameMoveType]] = []
        self.evalqueue: list[tuple[int, torch.Tensor, int]] = []
        self.respondqueue: list[tuple[int, torch.Tensor, int]] = []
        self.prob: torch.Tensor = torch.Tensor([])
        self.v: torch.Tensor = self.prob
        self.old_prob: torch.Tensor | None = None
        self.old_v: torch.Tensor | None = None
        self.isFinished: list[bool] = []
        self.finished = 0
        assert shards > 0
        self.shards = 1

    def schdule_eval(self, identifier: int, toSchedule: torch.Tensor, localindex: int):
        self.evalqueue.append((identifier, toSchedule, localindex))

    def update_cpuct(self, cpuct:float):
        for game in self.games:
            game.change_cpuct(cpuct)

    def check_finish_status(self, idx: int):
        if self.isFinished[idx]:
            return
        if self.games[idx].is_finished():
            self.finished += 1
            self.isFinished[idx] = True

    def add_game(self, game: MCTSForestParticipant[gameStateType, gameMoveType]):
        game.set_forest_identifier(len(self.games))
        self.games.append(game)
        self.isFinished.append(False)
        self.check_finish_status(len(self.games) - 1)
        self.resolve_queue()

    def execute_queue(self):
        if len(self.evalqueue) == 0:
            return
        self.old_prob, self.old_v = self.prob.detach().cpu(), self.v.detach().cpu()
        with torch.inference_mode():
            self.prob, self.v = self.model(torch.stack([tensor for _, tensor, _ in self.evalqueue]).to(self.device))
        self.resolve_queue()
        self.respondqueue = self.evalqueue
        self.evalqueue = []
    def resolve_queue(self):
        if self.old_prob is None:
            self.old_prob, self.old_v = self.prob.detach().cpu(), self.v.detach().cpu()
        for idx, (gameid, input, internal_id) in enumerate(self.respondqueue):
            self.games[gameid].postevaluate(input, self.old_prob[idx], self.old_v[idx], internal_id)
        self.respondqueue.clear()
        self.old_prob, self.old_v = None, None
    def execute_iterations(self, num_iters: int = 1):
        self.execute_queue()
        self.resolve_queue()
        shardsize = math.ceil(len(self.games) / self.shards)
        for _ in range(num_iters):
            for shard in range(self.shards):
                for idx in range(shardsize*shard, min(shardsize*(shard+1),len(self.games))):
                    self.check_finish_status(idx)
                    if not self.isFinished[idx]:
                        self.games[idx].perform_iteration()
                self.execute_queue()
            self.resolve_queue()

    def step_forward(self, temp: float = 1):
        self.execute_queue()
        self.resolve_queue()
        for idx, game in enumerate(self.games):
            self.check_finish_status(idx)
            if not self.isFinished[idx]:
                game.step(game.select(temp))

    def are_all_finished(self):
        return self.finished == len(self.games)

    def clear(self):
        self.games.clear()
        self.isFinished.clear()
        self.evalqueue.clear()
        self.finished = 0

    def select_in_game(self, idx: int, temp: float = 0) -> int | None:
        self.execute_queue()
        self.resolve_queue()
        self.check_finish_status(idx)
        if not self.isFinished[idx]:
            return self.games[idx].select(temp)

    def make_game_move(self, idx: int, move: int):
        self.execute_queue()
        self.resolve_queue()
        self.games[idx].step(move)
        self.check_finish_status(idx)


class MCTSTickTacToe(MCTSForestParticipant[GameState, int]):
    def __init__(self, model: nn.Module, device: torch.device, forest: MCTSForest[GameState, int] | None = None,
                 start_state: GameState | None = None, history_size: int = 1, cpuct: float = 1,
                 training_mode: bool = False, epsilon: float = 0, training_data_temp: float = 1.0, posteval_temp: float = 1.0):
        self.model, self.device = model, device
        self.forest = forest
        self.pathSave: list[list[tuple[int, int]]] = []
        self.currentNode: list[int] = []
        self._forest_identifier: int | None = None
        if start_state is None:
            start_state = GameState(True, -1, np.zeros(81, dtype=np.int8), np.full(9, -2), False)
        self.posteval_temp = posteval_temp
        super().__init__(start_state=start_state, history_size=history_size, cpuct=cpuct, training_mode=training_mode, epsilon=epsilon, training_data_temp=training_data_temp)

    def _apply_move(self, gameState: GameState, move: int) -> GameState:
        gameState = GameState(gameState.crossNowPlaying, gameState.nowallowed, gameState.board.copy(),
                              gameState.wins.copy(), gameState.finished)
        assert gameState.board[move] == 0
        gameState.board[move] = 1 if gameState.crossNowPlaying else 2
        gameState.crossNowPlaying = not gameState.crossNowPlaying
        subboard, submove = (move // 27) * 3 + (move%9) // 3, ((move // 9) % 3) * 3 + move % 3
        subboard_base = (subboard // 3) * 27 + (subboard % 3) * 3
        subboard_array = np.concat(
            [gameState.board[subboard_base: subboard_base + 3], gameState.board[subboard_base + 9:subboard_base + 12],
             gameState.board[subboard_base + 18:subboard_base + 21]])
        gameState.wins[subboard] = self._score_subboard(subboard_array)
        gameState.nowallowed = submove if gameState.wins[submove] == -2 else -1
        gameState.finished = self._score_game(gameState.wins) > -2
        return gameState

    def _score_line(self, a1: int, a2: int, a3: int):
        if a1 == a2 and a2 == a3:
            if a1 == 1: return 1
            if a1 == 2 or a1 == -1: return -1
        return 0

    def _score_game(self, winsBoard: np.ndarray) -> int:
        score = winsBoard.copy()
        score[score == -2] = 0

        res = self._score_subboard(score)
        if not res == -2:
            return res
        return -2 if min(winsBoard) == -2 else 0

    # returns -2 if game is unfinished, -1 if O is victorius, 0 for draw, and 1 if X is victorius
    def _score_subboard(self, board: np.ndarray):
        for i in range(0,7,3):
            r = self._score_line(board[i], board[i + 1], board[i + 2])
            if not r == 0:
                return r
        for i in range(3):
            r = self._score_line(board[i], board[i + 3], board[i + 6])
            if not r == 0:
                return r
        r = self._score_line(board[0], board[4], board[8])
        if not r == 0:
            return r
        r = self._score_line(board[2], board[4], board[6])
        if not r == 0:
            return r
        for i in board:
            if i == 0:
                return -2
        return 0

    def _hash_position(self, gamestate: GameState) -> int:
        hash = int(0)
        for v in gamestate.board:
            hash *= 3
            hash += int(v)
        hash *= 10
        hash += gamestate.nowallowed + 1
        return hash

    def get_possible_moves(self, gameState: GameState) -> list[int]:
        moveset: list[int] = []
        if gameState.finished:
            return moveset
        if gameState.nowallowed >= 0 and gameState.wins[gameState.nowallowed] == -2:
            bx, by = gameState.nowallowed % 3 * 3, (gameState.nowallowed // 3) * 3
            moveset = [(by + i) * 9 + bx + j for i in range(3) for j in range(3) if
                       gameState.board[(by + i) * 9 + bx + j] == 0]
        else:
            moveset = [i for i in range(9 * 9) if gameState.board[i] == 0]
        return moveset

    def evaluate(self, node: int, path_from_root: list[tuple[int, int]]):
        currentNode = self.transpositionTable[node]
        winsScore = self._score_game(currentNode.gameState.wins)
        if winsScore != -2:
            currentNode.eval = winsScore if currentNode.gameState.crossNowPlaying else -winsScore
            self._backprop(path_from_root, currentNode.eval)
            return
        self.pathSave.append(path_from_root)
        self.currentNode.append(node)
        if self._forest_identifier is None or self.forest is None:
            prop: torch.Tensor
            v: torch.Tensor
            embeded = currentNode.gameState.embed_board()
            with torch.inference_mode():
                prop, v = self.model(embeded.unsqueeze(0).to(self.device))
            self.postevaluate(embeded, prop.reshape(81), v[0], len(self.currentNode) - 1)
        else:
            self.forest.schdule_eval(self._forest_identifier, currentNode.gameState.embed_board(),
                                     len(self.currentNode) - 1)

    def postevaluate(self, input: torch.Tensor, prop: torch.Tensor, v: torch.Tensor, id: int):
        currentNode = self.transpositionTable[self.currentNode[id]]
        value = v.item()
        movemask = input[2].reshape(81)
        currentNode.eval = value
        currentNode.possibleMovesPropabilityDist = F.softmax(prop[movemask == 1]/self.posteval_temp,
                                                             dim=0).cpu().numpy().tolist()
        self._backprop(self.pathSave[id], value)
        self.pathSave.pop(id)
        self.currentNode.pop(id)

    def is_finished(self) -> bool:
        result = self._score_game(self.transpositionTable[self.rootnode].gameState.wins)
        return result != -2

    def game_result(self) -> int:
        assert self.is_finished()
        return self._score_game(self.transpositionTable[self.rootnode].gameState.wins)

    def get_training_data(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        assert self.is_finished() and self.training_mode
        result: int = self.game_result()
        data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for datasample in self.training_data:
            board = datasample[0].embed_board()
            props = torch.zeros((9, 9))
            counter = 0
            for x in range(9):
                for y in range(9):
                    if board[2][x][y] == 1:
                        props[x][y] = datasample[1][counter]
                        counter += 1
            data.append((board, props.float(), torch.tensor(result if datasample[0].crossNowPlaying else -result)))
        return data
    def prune(self, rootNode: MCTSNode[gameStateType, gameMoveType]):
        idx = np.where(rootNode.gameState.board != self.transpositionTable[self.rootnode].gameState.board)[0]
        ele = self.transpositionTable[self.rootnode].gameState.board[idx]
        for key in list(self.transpositionTable.keys()):
            targetboard = self.transpositionTable[key].gameState.board
            if targetboard[idx] != ele:
                self.transpositionTable.pop(key)


class Tournament():
    def __init__(self, model1: nn.Module, model2: nn.Module, device: torch.device):
        self.model1 = model1
        self.model2 = model2
        self.device = device

    def execute_play(self, forests: tuple[MCTSForest[GameState, int], MCTSForest[GameState, int]], normal: bool,
                     num_games: int, mctsiters: int, start_moves: np.ndarray|None = None):
        forest1, forest2 = forests
        for _ in range(num_games):
            forest1.add_game(MCTSTickTacToe(self.model1, self.device, forest1))
            forest2.add_game(MCTSTickTacToe(self.model1, self.device, forest2))
        if start_moves is not None:
            for f in range(start_moves.shape[0]):
                for idx in range(num_games):
                    forest1.make_game_move(idx, start_moves[f,idx])
                    forest2.make_game_move(idx, start_moves[f,idx])
        counter = 0
        while not forest1.are_all_finished():
            if normal:
                forest1.execute_iterations(mctsiters)
            else:
                forest2.execute_iterations(mctsiters)
            for idx in range(num_games):
                if normal:
                    selected = forest1.select_in_game(idx, 1 if counter < 10 else 0)
                else:
                    selected = forest2.select_in_game(idx, 1 if counter < 10 else 0)
                if selected is not None:
                    forest1.make_game_move(idx, selected)
                    forest2.make_game_move(idx, selected)
            normal = not normal
            print(f"played move nr {counter}")
            counter += 1

    def play(self, num_games: int, mctsiters: int):
        forest1, forest2 = MCTSForest[GameState, int](self.model1, self.device), MCTSForest[GameState, int](self.model2,
                                                                                                            self.device)
        start_move_1 = np.random.randint(0,81, size=num_games)
        start_move_2 = np.random.randint(0,8, size=num_games)
        self.execute_play((forest1, forest2), True, num_games, mctsiters,np.stack([start_move_1, start_move_2]))
        result: list[int] = [game.game_result() for game in forest1.games]
        forest1.clear()
        forest2.clear()
        self.execute_play((forest1, forest2), False, num_games, mctsiters, np.stack([start_move_1, start_move_2]))
        for game in forest1.games:
            result.append(-game.game_result())
        return result
