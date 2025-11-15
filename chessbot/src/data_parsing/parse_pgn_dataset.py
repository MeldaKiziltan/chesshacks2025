import chess
import chess.pgn
import sys
import os
import glob
import time

PRINT_DOT_EVERY_N_POSITIONS = 500000

def parse_game_fast(game):
    """
    Parses a single game and yields all (position, move) pairs.
    No ELO check needed, as the data is pre-filtered.
    """
    try:
        board = game.board()
        for move in game.mainline_moves():
            position_fen = board.fen()
            move_uci = move.uci()
            yield (position_fen, move_uci)
            board.push(move)
    except Exception:
        # skipping any weird/corrupted games
        pass

def main():
    print("--- Starting PGN Parse ---", file=sys.stderr)
    
    # total timer for the whole script
    total_start_time = time.monotonic()
    
    # glob to find all .pgn files in the current dir
    pgn_files = glob.glob("*.pgn")

    if not pgn_files:
        print("ERROR: No .pgn files found!", file=sys.stderr)
        print("Please make sure you have unzipped your files into this folder.", file=sys.stderr)
        return

    print(f"Found {len(pgn_files)} PGN files to parse:", file=sys.stderr)
    for f in pgn_files:
        print(f"  - {f}", file=sys.stderr)
    
    total_position_count = 0
    
    # final combined dataset file
    with open("training_data.txt", "w") as out_f:
        # looping through each PGN file
        for pgn_file in pgn_files:
            print(f"Parsing '{pgn_file}'...", file=sys.stderr)
            
            print("Progress: ", end='', file=sys.stderr)
            sys.stderr.flush() # Force it to print now
            
            # timer for each specific file
            file_start_time = time.monotonic()
            file_position_count = 0

            # open single pgn file
            with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    # reading one game at a time
                    try:
                        game = chess.pgn.read_game(f)
                    except Exception:
                        # encoding errors in messy PGNs ?
                        continue

                    if game is None:
                        # end of file
                        break
                    
                    # parsing game and write to our one big file
                    for (position_fen, move_uci) in parse_game_fast(game):
                        file_position_count += 1
                        out_f.write(f"{position_fen}|{move_uci}\n")
                        
                        if file_position_count % PRINT_DOT_EVERY_N_POSITIONS == 0:
                            print('.', end='', file=sys.stderr)
                            sys.stderr.flush() # Force the dot to print
            
            print("\nDone.", file=sys.stderr)
            
            # stop timer and print results
            file_end_time = time.monotonic()
            elapsed_seconds = file_end_time - file_start_time
            total_position_count += file_position_count
            
            print(f"Finished '{pgn_file}': Extracted {file_position_count} positions in {elapsed_seconds:.2f} seconds.", file=sys.stderr)

    # stop total timer, print final time and results
    total_end_time = time.monotonic()
    total_elapsed = total_end_time - total_start_time
    
    print("--- PGN Parsing Finished ---", file=sys.stderr)
    print(f"Total elite positions extracted: {total_position_count}", file=sys.stderr)
    print(f"Total time: {total_elapsed:.2f} seconds.", file=sys.stderr)
    print("Your dataset is ready in 'training_data.txt'")

if __name__ == "__main__":
    main()