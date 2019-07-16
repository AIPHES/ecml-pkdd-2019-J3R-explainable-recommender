import onmt
import onmt.io
import onmt.translate
import onmt.ModelConstructor
from collections import namedtuple
import numpy as np


def get_weights(args):
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

                
            # Load the model.
            Opt = namedtuple('Opt', ['model', 'data_type', 'reuse_copy_attn', "gpu"])

            opt = Opt(model_path, "text", False, 0)
            fields, model, model_opt =  onmt.ModelConstructor.load_test_model(opt,{"reuse_copy_attn":False})

                
            # Test data
            data = onmt.io.build_dataset(fields, "text", src_path, None, use_filter_pred=False)
            data_iter = onmt.io.OrderedIterator(
                    dataset=data, device=0,
                    batch_size=1, train=False, sort=False,
                    sort_within_batch=True, shuffle=False)
            # Translator
            translator = onmt.translate.Translator(model, fields,
                                                       beam_size=5,
                                                       n_best=1,
                                                       global_scorer=onmt.translate.GNMTGlobalScorer(0, 0, "none", "none"),
                                                       cuda=True)

            builder = onmt.translate.TranslationBuilder(
                    data, translator.fields,
                    1, False, None)

            for j, batch in enumerate(data_iter):
                    batch_data = translator.translate_batch(batch, data)
                    translations = builder.from_batch(batch_data)
                    print("src:", " ".join(translations[0].src_raw))
                    print("tgt:", " ".join(translations[0].pred_sents[0]))
                    translations[0].log(j)
                    np.save("./visualize/attns/" + "_".join(translations[0].src_raw) + ".npy", (translations[0].attns[0]).cpu().numpy())
                    print()

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
                


