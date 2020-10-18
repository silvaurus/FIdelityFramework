# Written by Yi He
# RTL fault injection 
# Support injecting to 1 bit of FF errors 

import argparse
import re
import os
import sys
import datetime
import time

# Given a register value, and a inject bit, return the force value of the register
def get_force(reg_val, sel_bit_set):
    force_val = ''
    for n_bit in range(len(reg_val)):
        if n_bit not in sel_bit_set:
            force_val = reg_val[len(reg_val) - n_bit - 1] + force_val
        else:
            if reg_val[len(reg_val) - n_bit - 1] == '0':
                force_val = '1' + force_val
            else:
                force_val = '0' + force_val
    return force_val

# Run one injection
def run_one(args):
    # Read input
    target = open(args.inj_info).readlines()[1]
    target = target.split()
    reg_name = target[0]
    inj_cycle = int(target[1])
    inj_bit = int(target[2])
    reg_val = target[3]
    force_val = get_force(reg_val, [inj_bit])
    target_set = [(reg_name, force_val, inj_cycle, [inj_bit])]

    print(target_set)
    print(len(target_set))

    print('Start running injection.')

    src_cwd = os.getcwd()
    if src_cwd[src_cwd.rfind('/')+1:] != 'injection':
        print('Your are running the script in a wrong directory.')
        print('Please go to injection directory.')
        exit()

    now = datetime.datetime.now()
    file_dir = os.getcwd()
    date = str(now.year) + '_' + str(now.month) + '_' + str(now.day)
    
    # For each date we have a folder, otherwise the total data would get too large to generate report!
    if not os.path.isdir('../data/' + date):
        os.system('mkdir -p ../data/' + date)
    index = 1
    directory = re.sub('\.s','',args.program) +'_inj_' + date + '_' + str(index)
    while os.path.isdir('../data/' + date + '/' + directory) is True:
        index += 1
        directory = re.sub('\.s','',args.program) + '_inj_' + date + '_' + str(index)
    os.system('mkdir ../data/' + date + '/' + directory)

    # Write injection testbench
    orig_bench = 'tb_inject_origin.v'
    original_testbench = open(orig_bench, 'r').read().splitlines()
    injection_testbench = open('../data/' + date + '/' + directory + '/tb_inject.v', 'w')

    for line in original_testbench:
        injection_testbench.write(line + '\n')
        words = line.split()

        # Define the dump file path
        #if len(words) == 3 and words[0] == '//' and words[1] == 'File' and words[2] == 'Path':
        #    injection_testbench.write('   $vcdplusfile(\"' + os.getcwd() + '/../data/' + date + '/' + directory + '/result.vpd\");\n');
        # Define the input vectors
        if len(words) == 4 and words[0] == '//' and words[1] =='Define' and words[2] == 'Input' and words[3] == 'Vector':
            for num in range(len(target_set)):
                injection_testbench.write('`define INPUT_REG_NAME_{} {}\n'.format(num, target_set[num][0]))
                injection_testbench.write('// For this reg inject to bits ')
                for val in target_set[num][3]:
                    injection_testbench.write('{}, '.format(val))
                injection_testbench.write('\n')
        # Write force statement
        if len(words) == 3 and words[0] == '//' and words[1] =='Insert' and words[2] == 'Force':
            inj_cycle = target_set[0][2]
            for num in range(len(target_set)):
                print(target_set[num])
                injection_testbench.write('  if (top.nvdla_top.counter == {}) begin\n'.format(inj_cycle))
                # Determine the force value
                injection_testbench.write('    force `INPUT_REG_NAME_{} = {}\'b{};\n'.format(num, len(target_set[num][1]), target_set[num][1]))
                injection_testbench.write('  end\n')
                injection_testbench.write('  else begin\n')
                injection_testbench.write('    release `INPUT_REG_NAME_{};\n'.format(num))
                injection_testbench.write('  end\n')

    injection_testbench.close()

    # Copy Makefile to directory
    os.system('cp Makefile_origin ../data/' + date + '/' + directory + '/Makefile')
    # Copy dut file to direcrtory
    os.system('cp {}/verif/dut/dut.f ../data/{}/{}/dut.f'.format(args.hw_path, date, directory))
    # Change to the testing directory
    os.chdir('../data/' + date + '/' + directory)

    # Run make build
    os.system('make build')
    # Sleep 30 seconds
    time.sleep(30)
    # Run make run
    if 'golden' in os.getcwd():
        os.system('make run TESTDIR=' + args.hw_path + '/verif/traces/traceplayer/' + args.program)
    else:
        os.system('make run TESTDIR=' + args.hw_path + '/verif/traces/traceplayer/' + args.program)
    # Compare vpd file
    #os.system('vcdiff -allsigdiffs -limitdiffs 0 -ignorewires ' + args.program + '/vcdplus.vpd ../../../golden/' + args.program + '/' + args.program + '/vcdplus.vpd > diff.txt')

    # Switch back to source directory
    os.chdir(src_cwd)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inj_info', required=True, help="The file that record the inject information")
    parser.add_argument('--program', required=True, help="The inject program")
    parser.add_argument('--hw_path', required=True, help="The path to NVDLA")

    args = parser.parse_args()
    run_one(args)


if __name__ == '__main__':
    main()
