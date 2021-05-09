"""
Model wrapepd in ctn_benchmark
"""

import nengo
import numpy as np
import nengo_spa as spa

from nengo_spa import sym
from nengo_spa.networks.selection import IA

from utils import DictMap
from xsetup import Feedback

import ctn_benchmark

class WCSTModel(ctn_benchmark.Benchmark):
    def params(self):
        # Experimental parameters
        self.default("Number of correct responses defining a category", x_seq_correct=3)
        self.default("Length of response", x_timesteps=300)
        self.default("Number of cards in the stimlus deck", x_deck_size=128)
        self.default("Use random rule sequence", x_random_rule=True)
        
        # Model parameters
        self.default("model seed", mseed=0)
        self.default("SP dim", d=512)
        self.default("feedback connection on rule memory", feedback_rule_strength=0.6)
        self.default("feedback connection rule synapse", feedback_rule_synapse=0.1)
        self.default("feedback connection on gate memory", feedback_gate_strength=0.9)
        self.default("number of neurons per WTA ensemble", wta_n_neurons=50)
        self.default("number of neurons per IA ensemble", ia_n_neurons=200)
        self.default("accumulation threshold IA", ia_accum_threshold=0.7)
        self.default("accumulation timescale IA", ia_accum_timescale=0.2)
        self.default("accumulation synapse IA", ia_accum_synapse=0.05)    
        self.default("connection ia to rule memory synapse", ia_to_rule_guess_synapse=0.15)
        self.default("compare module neurons per dimension", cmp_neurons_per_dimension=500)
        
        self.default("simulation length", T=5)
        self.default("direct mode", direct_mode=False)
        self.default("save probe data", save_probes=False)
        
        self.default("context", context=None)
        
        self.hidden_params.append("context")   

    def model(self, p):
        rng = np.random.RandomState(p.mseed)
        
        self.experiment = Feedback(
            seq_correct=p.x_seq_correct,          
            timesteps=p.x_timesteps,
            deck_size=p.x_deck_size,
            random_rule=p.x_random_rule,
            rng=rng)
        
        self.vocabs = DictMap()
        
        model = spa.Network(seed=p.mseed)

        with model:
            vocab = spa.Vocabulary(dimensions=p.d, pointer_gen=rng)
            vocab.populate(
                """
                V0; V1; V2; V3;
                R; B; G; Y;
                SQ; CR; ST; TR;
                ONE; TWO; THREE; FOUR;
                Color; Shape; Number; 
                C1={}; C2={}; C3={}; C4={}
                """.format(
                    self.experiment.target_cards_sps['C1'],
                    self.experiment.target_cards_sps['C2'],
                    self.experiment.target_cards_sps['C3'],
                    self.experiment.target_cards_sps['C4'],
                ))

            self.vocabs.choice = vocab.create_subset(['C1', 'C2', 'C3', 'C4']) 
            features = ['Number', 'Color', 'Shape']

            targets_sp = spa.Transcode(
                '(V0*C1 + V1*C2 + V2*C3 + V3*C4).normalized()',
                output_vocab=vocab, label='target_cards')

            vocab_feedback = spa.Vocabulary(dimensions=p.d, pointer_gen=rng)
            vocab_feedback.populate('Correct; Wrong; Neutral')

            feedback = spa.Transcode(
                self.experiment.step, 
                input_vocab=self.vocabs.choice,     # C1, C2, C3, C4
                output_vocab=vocab_feedback,  # Correct / Wrong / Neutral
                label='feedback')

            # initial guess at the rule
            init_guess = features[rng.randint(len(features))]        
            input_guess = spa.Transcode(
                lambda t: init_guess if 0 < t < .15 else '0', 
                output_vocab=vocab, label='input')

            guess = spa.State(
                vocab=vocab, 
                feedback=p.feedback_rule_strength,
                feedback_synapse=p.feedback_rule_synapse, 
                label='rule guess')
            nengo.Connection(input_guess.output, guess.input)

            stimulus_card = spa.Transcode(
                self.experiment.set_stimulus,
                output_vocab=vocab,
                label='stimulus card')

            wta = spa.networks.selection.WTA(n_neurons=p.wta_n_neurons,
                                             n_ensembles=4,
                                             threshold=0.0,
                                             function=lambda x: 1 if x>0 else 0)

            cue_guess = spa.modules.WTAAssocMem(
                input_vocab=vocab, threshold=0.3, mapping=vocab.keys())

            stimulus_card * ~guess >> cue_guess

            # Choice is the selected card (doesn't like being a state)
            choice = spa.modules.WTAAssocMem(
                input_vocab=self.vocabs.choice,
                mapping=self.vocabs.choice.keys(),
                threshold=0.2,
                label='choice')

            for i in range(4):
                dot = spa.Compare(
                    vocab, neurons_per_dimension=p.cmp_neurons_per_dimension, label='dot {}'.format(i))
                ((targets_sp * ~getattr(sym, 'V%d'% i)) * ~guess) >> dot.input_a
                cue_guess >> dot.input_b

                # WTA scalar values for each target card
                nengo.Connection(dot.output, wta.input[i])
                # `choice` is one of (C{1,2,3,4}) SPs
                nengo.Connection(wta.output[i], choice.input,
                                 transform=np.expand_dims(
                                     vocab['C%d'%(i+1)].v, axis=1))

            choice >> feedback

            bgout = spa.WTAAssocMem(
                threshold=0.7, input_vocab=vocab, mapping=vocab.keys(),
                label='bgout')
            # changing for new sims
            nengo.Connection(bgout.output, bgout.input, transform=p.feedback_gate_strength) #.9

            factor_fb = 0.5
            with spa.ActionSelection() as action_sel:
                spa.ifmax(
                    factor_fb*spa.dot(feedback, sym.Neutral) + (1-factor_fb)*spa.dot(feedback, sym.Correct),
                    guess >> bgout,
                    ) 
                spa.ifmax(
                    factor_fb * spa.dot(feedback, sym.Wrong) + (1-factor_fb) * spa.dot(guess, sym.Shape),
                    sym.Color >> bgout)
                spa.ifmax(
                    factor_fb * spa.dot(feedback, sym.Wrong) + (1-factor_fb) * spa.dot(guess, sym.Color),
                    sym.Number >> bgout)            
                spa.ifmax(
                    factor_fb * spa.dot(feedback, sym.Wrong) + (1-factor_fb) * spa.dot(guess, sym.Number),
                    sym.Shape >> bgout)         


            self.vocabs.gate = vocab.create_subset(features)
            # accum* parameters below seem to be important for sequential selection 
            #  rules, otherwise rules overlap
            gate = IA(n_neurons=p.ia_n_neurons, n_ensembles=3,
                      accum_threshold=p.ia_accum_threshold,
                      accum_timescale=p.ia_accum_timescale,
                      accum_synapse=p.ia_accum_synapse)
            nengo.Connection(bgout.output, gate.input, transform=self.vocabs.gate.vectors)
            nengo.Connection(gate.output, guess.input, transform=.5*self.vocabs.gate.vectors.T,
                             synapse=p.ia_to_rule_guess_synapse)  #0.15    
            nengo.Connection(feedback.output, gate.input_reset,
                             transform=np.expand_dims(vocab_feedback['Correct'].v, axis=1).T) 

            if p.save_probes:
                self.p_gate = nengo.Probe(gate.output, synapse=0.01)
                self.p_guess = nengo.Probe(guess.output, synapse=0.01)
                self.p_choice = nengo.Probe(choice.output, synapse=0.01)
                self.p_bgout = nengo.Probe(bgout.output, synapse=0.01)
                self.p_fb = nengo.Probe(feedback.output, synapse=0.01)
                self.p_th = nengo.Probe(action_sel.thalamus.output, synapse=0.01)
            
            if p.direct_mode:
                print("Running spa.{Bind,Compare} in direct mode!")
                for net in model.networks:
                    if isinstance(net, (spa.Bind, spa.Compare)):
                        for ens in net.all_ensembles:
                            ens.neuron_type = nengo.Direct()
        return model

    def evaluate(self, p, sim, plt):
        sim.run(p.T)
        return self.experiment.resp_sheet
    
if __name__=="__main__":

    model = WCSTModel().make_model(
        x_seq_correct=10,
        x_timesteps=100,
        x_deck_size=64,
        x_random_rule=True,
        save_probes=True,
        T=10,
        d=64)
   