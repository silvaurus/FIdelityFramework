#!/bin/python3
# Yi He
# University of Chicago
# Collect the RTL injection result

import os
import sys
import argparse

def collect_result(args):
    # Check the integrity of files
    tb_file_name = args.data_path + "/" + args.date + "/" + args.test + "/tb_inject.v"
    if not os.path.isfile(tb_file_name):
        print("ERROR: test bench not found!")
        exit(1)

    cmp_file_name = args.data_path + "/" + args.date + "/" + args.test + "/simv.compile.log"
    if not os.path.isfile(cmp_file_name):
        print("ERROR: compile log not found!")
        exit(1)

    test_file_name = args.data_path + "/" + args.date + "/" + args.test + "/" + args.test[:(args.test.rfind('inj')-1)] + "/test.log"  
    if not os.path.isfile(test_file_name):
        print("ERROR: test log not found!")
        exit(1)

    result_file_name = args.data_path + "/" + args.date + "/" + args.test + "/" + args.test[:(args.test.rfind('inj')-1)] + "/0.chiplib_replay.raw2"
    if not os.path.isfile(result_file_name):
        print("ERROR: test log not found!")
        exit(1)

    # Compare output feature map
    diff_output_indexes = []
    golden_diff_output = []
    inject_diff_output = []

    golden_file = open(args.golden_result, 'r').read().splitlines()
    test_file = open(result_file_name, 'r').read().splitlines()

    num_line = len(golden_file)
    num_diff = 0
    for i in range(0,num_line):
        golden_words = golden_file[i]
        test_words = test_file[i]
        if golden_words[:4] != test_words[:4] and test_words[:4] != "xxxx":
            num_diff += 1
            diff_output_indexes.append(2 * i)
            golden_diff_output.append(golden_words[:4])
            inject_diff_output.append(test_words[:4])
        if golden_words[4:] != test_words[4:] and test_words[4:] != "xxxx":
            num_diff += 1
            diff_output_indexes.append(2 * i + 1)
            golden_diff_output.append(golden_words[4:])
            inject_diff_output.append(test_words[4:])
    print("*********************************************")
    print("Fault injection result:")
    print("Number of different output neurons: {}".format(num_diff))
    print("Here are the mismatching neurons:")
    for i in range(len(diff_output_indexes)):
        print("Index: {}, golden value: {}, faulty value: {}".format(diff_output_indexes[i], golden_diff_output[i], inject_diff_output[i]))
    print("*********************************************")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help="The path to data")
    parser.add_argument('--date', required=True, help="The date")
    parser.add_argument('--test', required=True, help="The name of the test")
    parser.add_argument('--golden_result', required=True, help="Path to the golden result")

    args = parser.parse_args()
   
    collect_result(args) 

if __name__ == '__main__':
    main()
