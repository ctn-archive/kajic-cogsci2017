import nengo
import numpy as np
import os
import os.path
import pdb
import pytry

from nengo import spa
from nengo.utils import numpy as npext
from cogsci17_semflu import fan


class SemFlu(pytry.NengoTrial):
    def params(self):
        self.param('word vector dimensions', d=64)
        self.param('cue connection feedback strength', c_fs=.2)
        self.param('state connection feedback strength', s_fs=1.)
        self.param('cue to state connection strength', cs_s=3)
        self.param('simulation length', sim_len=5)
        self.param('association matrix', amat='ngram_mat')
        self.param('cue state synapse', cs_syn=0.005)
        self.param('response to response magnitude synapse', rspm_syn=0.005)
        self.param('wta to response synapse', wtar_syn=0.1)
        self.param('inhibitory connection', inh_st=-5)
        self.param('wta threshold', wta_th=0.3)

        self.param('record and save spikes to file', save_spikes='')

    def model(self, p):
        d = p.d
        c_fs = p.c_fs

        data_dir = os.path.join(
            os.path.dirname(__file__),
            os.pardir, os.pardir, 'association_data')

        with spa.SPA(seed=p.seed) as model:

            # Load association data
            assoc_mat, i2w, _ = fan.load_assoc_mat(data_dir, p.amat)
            i2w = [i.upper() for i in i2w]

            # Create vectors
            self.vocab = fan.gen_spa_vocab(dimensions=d, word_list=i2w)

            vocab2 = self.vocab.create_subset(i2w)

            # Transformation matrix
            tr = np.dot(self.vocab.vectors.T,
                        np.dot(assoc_mat.T, self.vocab.vectors))

            # Cue ensemble
            model.cue = spa.State(
                vocab=self.vocab, dimensions=d, feedback=c_fs)

            # State ensemble
            model.state = spa.State(
                vocab=vocab2, dimensions=d, feedback=p.s_fs)

            nengo.Connection(
                model.cue.output, model.state.input, transform=p.cs_s*tr,
                synapse=p.cs_syn)

            model.wta = spa.AssociativeMemory(
                input_vocab=vocab2, output_vocab=vocab2, wta_output=True,
                wta_synapse=0.01, threshold=p.wta_th)

            model.response = spa.AssociativeMemory(
                input_vocab=self.vocab, wta_output=True)

            nengo.Connection(
                model.wta.output, model.response.input, synapse=p.wtar_syn,
                transform=3)

            nengo.Connection(model.state.output, model.wta.input)

            model.response_magnitude = spa.State(1)
            nengo.Connection(model.response.am.elem_output,
                             model.response_magnitude.input,
                             transform=np.ones((1, model.response.am.elem_output.size_out)),
                             synapse=p.rspm_syn)

            model.goal = spa.State(16)

            model.used_words = spa.State(
                vocab=vocab2, dimensions=d, feedback=1.)

            # inhibitory connection, prevents words from appearing again
            nengo.Connection(
                model.used_words.output, model.state.input, transform=p.inh_st)

            actions = spa.Actions(
                'dot(goal, INIT) --> cue=ANIMAL, goal=THINK',
                'dot(goal, THINK) + response_magnitude - 1 --> ' +
                    'cue=response, used_words=response, goal=THINK',
                '0.4 --> cue=ANIMAL, goal=THINK'
                )
            model.bg = spa.BasalGanglia(actions)
            model.thal = spa.Thalamus(model.bg)

            model.input = spa.Input(goal=lambda t: 'INIT' if t < 0.05 else '0')

            self.probe_response = nengo.Probe(
                model.response.output, synapse=0.03)

            for obj in model.all_objects:
                rng = np.random.RandomState(p.seed)
                if obj.seed is None:
                    obj.seed = rng.randint(npext.maxint)

            if p.save_spikes != '':
                self.p_cue = nengo.Probe(model.cue.output, synapse=0.03)
                self.p_cue_spikes = [
                    nengo.Probe(e.neurons, 'spikes')
                    for e in model.cue.state_ensembles.ensembles]
                self.p_bg_gpi_spikes = [
                    nengo.Probe(e.neurons, 'spikes')
                    for e in model.bg.gpi.ensembles]

        return model

    def evaluate(self, p, sim, plt):
        with sim:
            sim.run(p.sim_len)

        if p.save_spikes != '':
            np.savez(
                p.save_spikes, 
                t=sim.trange(),
                cue_decoded=spa.similarity(sim.data[self.p_cue], self.vocab),
                cue=np.concatenate(
                    [sim.data[p] for p in self.p_cue_spikes], axis=1),
                bg_gpi=np.concatenate(
                    [sim.data[p] for p in self.p_bg_gpi_spikes], axis=1))

        min_sim = 0.8   # discard responses while model initializes
        similarities = spa.similarity(
            sim.data[self.probe_response], self.vocab)

        mid = np.argmax(similarities, axis=1)
        sim_max = np.max(similarities, axis=1)

        responses = mid[sim_max > min_sim]
        timing = 1000*sim.trange()[sim_max > min_sim]

        if len(responses) > 0:
            current = responses[0]
    
            out_responses = [self.vocab.keys[current].lower()]
            out_time = [timing[0]]
            last_time = timing[0]  # needed for computing irt
    
            for idx in range(len(responses)-1):
                if responses[idx] != responses[idx+1]:
                    current = responses[idx+1]
                    word = self.vocab.keys[current]
    
                    if word not in ['ANIMAL']:
                        delta_t = timing[idx+1]-last_time
                        out_time.append(delta_t)
                        out_responses.append(word.lower())
                        last_time = timing[idx+1]
        else:
            out_responses = []
            out_time = []

        return {
            'responses': out_responses,
            'irt': out_time
            }

if __name__ == '__builtin__':
    model = SemFlu().make_model(d=256, seed=12)
