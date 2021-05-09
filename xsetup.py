import pdb
import nengo_spa as spa
import numpy as np
import pandas as pd
from collections import OrderedDict, defaultdict

class Feedback():
    def __init__(self,
                 seq_correct=5,
                 timesteps=150,
                 deck_size=64,
                 stim_f='./exp-data/stimulus.csv',
                 target_f='./exp-data/targets.csv',
                 rules_f='./exp-data/rules.csv',
                 random_rule=False,
                 rng=None
                ):
        """
        stim_f:      path to the CSV file listing stimulus card features for each trial
        target_f:    path to the CSV file listing target card features, remain fixed for the experiment
        rules_f:     path to the CSV file listing rules to apply
        """                
        self.n_seq_correct = seq_correct  # Threshold for switching the rule
        self.deck_size = deck_size        # Number of stimulus cards
        self.dt = timesteps               # Feedback length (in timesteps), relevant for dynamics

        self.n_correct = 0                # Counter for correct responses given a rule
        self.trial = 0
        self.seed = 0
        self.total_correct = 0            # Total number of correct responses in an experiment     
        self.n_categories = 0             # Total # of categories (sequences of # n_seq_correct)

        self.current_rule_idx = 0         # Index into the current rule (circular index, mod len(nr_rules))
        self.previous_rule_idx = 0        # Index into the previous rule
        self.current_card_idx = 0         # Index onto the stimulus card 
        self.len_current = 0              # Counter for positive feedback (reset when reaching self.dt)
        self.change_stimulus = False      # Toggled when a correct choice was made
        self.hold_counter = 0             # Length (in timesteps) of the neutral feedback after 
                                          #   correct response        
        self.rng = rng
            
        self.resp_sheet = defaultdict(list)       
        self.card_template = '(Color*{} + Shape*{} + Number*{}).normalized()'
        
        # Load experimental data as pandas dataframes
        upper = lambda x: x.upper()
        self.stimulus_df = pd.read_csv(
            stim_f, index_col=0, converters={1: upper, 2: upper, 3: upper})
        
        # Pick the first or the random rule sequence
        self.random_rule = random_rule
        all_rules = pd.read_csv(rules_f, index_col=0)
        self.rule_seq_id = rng.randint(len(all_rules)) if self.random_rule else 0
        series = all_rules.T[self.rule_seq_id]
        series.name = "rule"
        self.rules_df = pd.DataFrame(series)
        
        self.targets_df = pd.read_csv(
            target_f, index_col=0, converters={1: upper, 2: upper, 3: upper})
        
        self.target_cards_sps = self._make_target_sps()          # C#n : Color*A + Number*B +...
        self.feature_to_target = self._make_feature_to_target()  # blue: c1, heart: c2....

    def __repr__(self):
        return "seq_correct: {}, timesteps: {}, decksize: {}, random rule: {}".format(
            self.n_seq_correct, self.dt, self.deck_size, self.random_rule)
    
    def _make_feature_to_target(self):
        """Return dictionary mapping features to cards, e.g., 'yellow': 'C1'."""
        feature_to_card = {}
        for x in self.targets_df.to_dict().values():
            cards = map(lambda y: 'C{}'.format(y), list(x.keys()))
            feature_to_card.update(dict(zip(x.values(), cards)))
        return feature_to_card
    
    def _make_target_sps(self):
        """Return SP description for the target cards."""
        return dict(zip(
            ['C{}'.format(i+1) for i in range(4)],
            list(self.targets_df.apply(lambda x: self.card_template.format(*x), axis=1).values)))
    
    @property
    def current_rule(self):
        return self.rules_df.iloc[self.current_rule_idx].rule
    
    @property
    def previous_rule(self):
        return self.rules_df.iloc[self.previous_rule_idx].rule   
    
    @property
    def response_sheet(self):
        """
        Return pandas dataframe response sheet for the whole experiment
        """
        return pd.DataFrame(self.resp_sheet)
    
    
    def update_current_rule(self):
        # circular index
        self.previous_rule_idx = self.current_rule_idx
        self.current_rule_idx = (self.current_rule_idx+1)%len(self.rules_df)

    def step(self, t, ptr):
        """Provide feedback in each step, feedback is one of: Neutral, Wrong, Correct"""
        # Initialization of the model, or just after the feedback
        if t < 0.2 or self.hold_counter > 0:
            self.hold_counter = max(0, self.hold_counter-1)
            return 'Neutral'

        current_rule = self.current_rule
        # the feature for the current stimulus card & the rule
        feature = self.stimulus_df.iloc[self.current_card_idx][current_rule] 
        
        # find target card that has that feature
        card = self.feature_to_target[feature]
        card_truth = int(card[-1]) - 1
        
        sim = spa.similarity(ptr, ptr.vocab.vectors, normalize=True)
        card_select = np.argmax(sim)
        
        # Similarity threshold for the card choice
        if sim[card_select] < 0.9:
            return 'Neutral'
        
        # Hold response for self.dt steps
        if self.len_current < self.dt:
            self.len_current += 1
            if card_truth != card_select:
                return 'Wrong'
            return 'Correct'

        self.trial += 1
        self.change_stimulus = True
        self.len_current = 0
        self.hold_counter = self.dt
        
        # Feedback "Incorrect"
        if card_truth != card_select:                        
            self.update_sheet('X', card, card_select, sim[card_select], t)
            self.n_correct = 0
            return 'Wrong'

        # Feedback "Correct"
        self.total_correct += 1
        self.n_correct += 1
        self.update_sheet(str(self.n_correct), card, card_select, sim[card_select], t)
        if self.n_correct == self.n_seq_correct:
            self.n_categories += 1 
            self.update_current_rule()
            self.n_correct = 0
        return 'Correct'
    
    def _find_match(self, s, t):
        matched = [self.feature_to_target[f] == t for f in s]
        return np.array(['C', 'S', 'N'])[np.array(matched)]
    
    def update_sheet(self, correct, target, choice, sim, t):
        """
        correct: X if incorrect, int for the n-th correct
        target: ground truth
        choice: selected card / card stimulus matched to
        sim: similarity between current card and ideal
        t: time
        """
        choice += 1  # card idx selected by model, count from 1
        stimulus_card = self.stimulus_df.iloc[self.current_card_idx].values
        match = self._find_match(stimulus_card, 'C' + str(choice))
        
        # self.resp_sheet['seed'].append(self.seed)
        self.resp_sheet['r_tstart'].append(round(t-self.dt/1000, 2))
        self.resp_sheet['r_tend'].append(round(t, 2))
        self.resp_sheet['trial'].append(self.trial)
        self.resp_sheet['match'].append(''.join(match))
        self.resp_sheet['stimulus'].append(
            '-'.join(stimulus_card))
        self.resp_sheet['target'].append(target[-1])
        self.resp_sheet['similarity'].append(sim)
        self.resp_sheet['choice'].append(choice)
        self.resp_sheet['rule'].append(self.current_rule[0].upper())
        self.resp_sheet['rule_seq_id'].append(self.rule_seq_id)
        self.resp_sheet['correct'].append(correct)
        self.resp_sheet['n_categories'].append(self.n_categories)
        self.resp_sheet['error'].append(int(correct=='X'))
               
        # Perseverative errors
        same_rule_guess = 0 
        if self.trial > 1:
            previous_guess = self.resp_sheet['match'][-2]           
            current_guess = self.resp_sheet['match'][-1]
            same_rule_guess = bool(set(current_guess).intersection(set(previous_guess)))            
        self.resp_sheet['p_error'].append(int(correct=='X' and same_rule_guess))
        
        # Perseverative response
        p_response = 0        
        if self.trial > 1 and self.resp_sheet['n_categories'][-1] > 0:
            previous_rule = self.previous_rule[0].upper()
            current_guess = self.resp_sheet['match'][-1]
            p_response = int(len(set(previous_rule).intersection(current_guess)))
        self.resp_sheet['p_response'].append(p_response and int(correct=='X'))
        
        # Failure to maintain set
        fail_shift = 0
        if self.trial > 1:
            prev_seq_correct = self.resp_sheet['correct'][-2]
            fail_shift = int(correct == 'X' and prev_seq_correct != 'X' and 4 < int(prev_seq_correct) < 10)
        self.resp_sheet['fail_shift'].append(fail_shift)
    
    def set_stimulus(self, t):
        """Set the new stimulus card."""
        if self.change_stimulus:
            self.current_card_idx = (self.current_card_idx+1)%len(self.stimulus_df)
            self.change_stimulus = False
        
        card_info = self.stimulus_df.iloc[self.current_card_idx]
        return self.card_template.format(*card_info)