import pandas as pd
import numpy as np
import os
import pytry

from cogsci17_semflu.process_responses import get_category_switches_heuristic
from cogsci17_semflu.fan import load_animal_categories


def process_output(data_path, nr_samp=30):
    animal_cat_path = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, 'animal_data',
        'animal_cat_dicts.pkl')
    ctoa, atoc = load_animal_categories(animal_cat_path)

    # load simulation data
    df = pd.DataFrame(pytry.read(data_path))

    # simulations without any responses
    no_resp = np.where(df.responses.apply(lambda x: len(x)).values < 1)[0]
    if len(no_resp) > 0:
        print('Simulations without no responses:', len(no_resp))
        df.drop(df.index[no_resp], inplace=True)

    # convert responses to lower-caps
    df['responses'] = df['responses'].apply(
        func=lambda x: [y.lower() for y in x])

    del df['backend']
    del df['d']
    del df['dt']
    del df['sim_len']
    del df['amat']
    del df['c_fs']
    columns = [u'sid', u'entry', u'irt', u'fpatchnum',
               u'fpatchitem', u'fitemsfromend',
               u'flastitem', u'meanirt', u'catitem']
    output = pd.DataFrame(columns=columns)

    for row_idx, row in df.iterrows():
        patch_num = 1
        responses = row.responses[:nr_samp]
        irts_row = row.irt[:nr_samp]
        t_cl, animals = get_category_switches_heuristic(responses, irts_row)

        counter = 0
        for ai, (a_c, t_c) in enumerate(zip(animals, t_cl)):
            for cai, animal in enumerate(a_c):
                counter += 1

                # make note if animal last in the cluster
                last = 0
                if animal == a_c[-1]:
                    last = 1

                # position in the cluster from the end
                fromend = len(a_c)-cai

                #  extract irt (computed automatically in the simulation)
                irt = t_c[cai]

                output = output.append([{'entry': animal, 'sid': int(row.seed),
                                         'fpatchnum': patch_num,
                                         'fpatchitem': cai+1,
                                         'fitemsfromend': fromend,
                                         'flastitem': last,
                                         'catitem': counter,
                                         'irt': irt, 'meanirt': 1}],
                                       ignore_index=True)
            patch_num += 1

    # Convert some bits to integers
    output['flastitem'] = output['flastitem'].apply(int)
    output['irt'] = output['irt'].apply(int)
    output['sid'] = output['sid'].apply(int)

    # Compute mean IRTs
    mean_irt = output.groupby('sid').mean().irt
    for sid, mirt in mean_irt.iteritems():
        output.set_value(output.sid == sid, 'meanirt', mirt)

    # Compute data on means and std. deviations
    sids = np.unique(output.sid)
    avgs = []
    for sid in sids:
        nritems = len(output[output.sid == sid])
        avgs.append(nritems)
    avgs = np.array(avgs)
    print('Mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}'.format(
        np.mean(avgs), np.std(avgs), np.min(avgs), np.max(avgs)))

    print("Average response length:", len(output)/(np.max(output.sid)+1))
    output.index += 1 # start with 1

    return output

if __name__ == "__main__":
    ou = process_output('./data/fan_tmp')
