import subprocess
import concurrent
from concurrent.futures import wait
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_proc', type=int, required=True)
    args = parser.parse_args()
    futures = []
    maj_perc_list = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875][::-1]
    n_class_list = [2, 4, 8, 16, 32, 64][::-1]
    args_tuple_list = []
    for perc in maj_perc_list:
        for n_c in n_class_list:
            args_tuple_list.append((perc, n_c))
    DATA_GEN_ENCRYPTED_PATH = "/home/zl310/cs585_project/vmoe/data_gen_encrypted.py"
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_proc) as executor:
        for idx, inputs in enumerate(args_tuple_list):
            command_list = ["python", DATA_GEN_ENCRYPTED_PATH,
                    "--train_or_test", str("train"),
                    "--maj_perc", str(inputs[0]),
                    "--n_classes", str(inputs[1])]
            futures.append(executor.submit(
                subprocess.run,command_list,
                capture_output=True, text=True))
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                data = future.result()
                print('\n\n' + str(idx) + '\n' +str(data).replace("\\n","\n") + '\n\n\n')
            except Exception as e:
                print(e)
        wait(futures)
