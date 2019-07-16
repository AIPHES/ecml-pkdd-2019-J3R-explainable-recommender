import argparse
import os
from subprocess import check_output

def mkdirp(path):
    """Checks if a path exists otherwise creates it
    Each line in the filename should contain a list of URLs separated by comma.
    Args:
        path: The path to check or create
    """
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def evaluate(args):
    corpora = ['music', 'cds', 'kindle', 'toys', 'yelp', 'electronics', 'tv']
    models = ['attention', 'pointer']
       
    for corpus in corpora:
        for model in models:
            models_path = "%s/%s/%s/" % (args.models_path, model, corpus)
            results_dir = args.results_path
            if not os.path.isdir(results_dir):
                mkdirp(results_dir)

            model_path = "/".join([models_path, "model_step_50000.pt"])
            if os.path.isfile(model_path):
                # Arguments for evaluation
                result_path = "%s/%s_%s_%s.txt" % (results_dir, model, corpus, "50000")
                output_path = "%s/%s/%s/%s" % (args.output_path, model, corpus, "output_50000_pred.txt")
                src_path = "%s/%s/test.src" % (args.data_dir, corpus)
                tgt_path = "%s/%s/test.tgt" % (args.data_dir, corpus)
                trans_command = "python translate.py -model %s -src %s -o %s -beam_size 10 -gpu 0 -replace_unk;" % (model_path, src_path, output_path)
                print(trans_command)
                """
                try:
                    output = check_output(trans_command, shell=True)
                except:
                    continue
                """
                scores_command = "files2rouge %s %s > %s" % (tgt_path, output_path, result_path)
                print(scores_command)
                try:
                    output = check_output(scores_command, shell=True)
                except:
                    continue
                #exit(0)


def get_args():
    ''' This function parses and return arguments passed in'''

    parser = argparse.ArgumentParser(description='Evaluate the Neural Summarization Models')

    #parser.add_argument('--summary_length', type=str, help='Summary Length ex:100', required=True)

    parser.add_argument('--data_dir', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/"))

    parser.add_argument('--models_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/"))

    parser.add_argument('--results_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/"))
    parser.add_argument('--output_path', type=str, help='Data Directory', required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/outputs/"))

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    evaluate(args)

