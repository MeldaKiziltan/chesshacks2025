# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 23:28:44 2025

@author: esha.pakalapati
"""

import random
import sys
import time
import os

INPUT_FILE = "training_data.txt"
TRAIN_FILE = "train_set.txt"
VAL_FILE = "validation_set.txt"

VAL_PERCENTAGE = 0.02 # 2% is a good start for a huge dataset i think

PRINT_DOT_EVERY_N_LINES = 2000000

print(f"--- Starting Data Split ---", file=sys.stderr)
print(f"Input: {INPUT_FILE} (Splitting {VAL_PERCENTAGE*100}% to {VAL_FILE})", file=sys.stderr)

total_start_time = time.monotonic()

try:
    with open(INPUT_FILE, "r") as f_in, \
         open(TRAIN_FILE, "w") as f_train, \
         open(VAL_FILE, "w") as f_val:
        
        train_count = 0
        val_count = 0
        line_count = 0
        
        print("Splitting data (this may take a while)...", file=sys.stderr)
        
        print("Progress: ", end='', file=sys.stderr) 
        sys.stderr.flush()
        
        # reading giant file line by line
        for line in f_in:
            # doing random to avoid bias from only one month or one game
            if random.random() < VAL_PERCENTAGE:
                f_val.write(line)
                val_count += 1
            else:
                f_train.write(line)
                train_count += 1
                
            line_count += 1
            if line_count % PRINT_DOT_EVERY_N_LINES == 0:
                print('.', end='', file=sys.stderr) # Print a dot with no newline
                sys.stderr.flush() # Force the dot to appear immediately

        print("\nDone.", file=sys.stderr) 

        total_end_time = time.monotonic()
        total_elapsed = total_end_time - total_start_time

        print("\n--- Split Finished ---", file=sys.stderr)
        print(f"Total time: {total_elapsed:.2f} seconds", file=sys.stderr)
        print(f"Training positions:   {train_count}", file=sys.stderr)
        print(f"Validation positions: {val_count}", file=sys.stderr)

except FileNotFoundError:
    print(f"ERROR: Input file not found: '{INPUT_FILE}'")
    print("Please make sure 'training_data.txt' is in the same folder.")