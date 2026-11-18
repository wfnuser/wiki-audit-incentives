"""Hand-derived cases protect IC signs, missing denominators and cluster sampling."""
import importlib
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest
import random
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class StatisticsTests(unittest.TestCase):
    def m(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.large_sample_statistics'))
        return importlib.import_module('experiments.large_sample_statistics')

    def rows(self):
        return [dict(id='a',task='T',source_hash='x',s=0,a=0),
                dict(id='b',task='T',source_hash='y',s=None,a=1),
                dict(id='c',task='T',source_hash='z',s=1,a=None)]

    def test_all_missing_completions_keep_assigned_denominator(self):
        got=self.m().completion_counts(self.rows())
        self.assertEqual(set(got),{(1,1,1,0),(1,1,0,1),(1,0,1,1),(1,0,0,2)})
        self.assertTrue(all(sum(x)==3 for x in got))

    def test_many_missing_rows_merge_equal_counts_without_dropping_cases(self):
        # Forty unknown binary observations have 2**40 assignments but only
        # 41 distinct sufficient-count vectors. A raw-world cap is incorrect.
        rows=[dict(id=str(i),task='T',source_hash=str(i),s=None,a=0) for i in range(40)]
        self.assertEqual(self.m().completion_counts(rows),[(i,0,40-i,0) for i in range(41)])

    def test_bootstrap_copies_of_one_unknown_row_share_one_completion(self):
        # A source drawn twice duplicates the same missing value; it does not
        # create two independently imputable values.
        got=self.m()._count_offsets(((2,(0,2)),))
        self.assertEqual(set(got),{(2,0,0,0),(0,0,2,0)})
        self.assertNotIn((1,0,1,0),got)

    def test_weighted_count_compression_matches_independent_assignment_oracle(self):
        m=self.m();rng=random.Random(8618)
        for _ in range(25):
            terms=tuple((rng.randrange(4),rng.choice(((0,1),(0,2),(1,3),(2,3),(0,1,2,3)))) for _ in range(6))
            expected=set()
            for cells in product(*(cs for _,cs in terms)):
                counts=[0]*4
                for (weight,_),cell in zip(terms,cells):counts[cell]+=weight
                expected.add(tuple(counts))
            self.assertEqual(set(m._count_offsets(terms)),expected)

    def test_sixty_source_result_is_recovered_from_literal_counts(self):
        v=self.m().law_metrics((22,2,30,6))
        self.assertAlmostEqual(v['h'],.6);self.assertAlmostEqual(v['a0'],2/15)
        self.assertAlmostEqual(v['a_plus'],1/6);self.assertAlmostEqual(v['delta'],.02)
        self.assertAlmostEqual(v['weak_min_reward'],130/3)
        self.assertAlmostEqual(v['target_credit'],-5237/2100)
        self.assertAlmostEqual(v['blind_credit'],-54641/15750)
        self.assertAlmostEqual(v['target_menu_margin'],-5237/2100)

    def test_undefined_condition_and_no_information_not_zero_success(self):
        m=self.m();v=m.law_metrics((3,1,0,0))
        self.assertIsNone(v['a_plus']);self.assertEqual(v['weak_min_reward'],float('inf'))
        self.assertIsNone(m.calibrate([(1,1,1,1)]))
        self.assertIsNone(m.calibrate([(0,1,0,1)]))

    def test_complete_menu_contract_reconstructs_old_missing_robust_solution(self):
        v=self.m().calibrate([(10,0,21,9),(10,0,20,10)])
        self.assertEqual(v['minimum_reward'],'1057/50');self.assertEqual(v['R'],'22')
        self.assertEqual(v['D'],'7757/1050')
        self.assertEqual(v['deposit_interval'],['551/75','52/7'])
        self.assertIsNone(self.m().calibrate([(10,0,21,9),(10,0,20,10)],radius=F(1,40)))

    def test_elimination_matches_independent_vertex_enumeration(self):
        from experiments.calibration_robustness import minimum_reward
        rng=random.Random(8617)
        for _ in range(40):
            counts=[tuple(rng.randrange(1,30) for _ in range(4)) for _ in range(2)]
            radius=rng.choice((F(0),F(1,1000),F(1,100)))
            laws=[tuple(F(v,sum(x)) for v in x) for x in counts]
            reference=minimum_reward(laws,radius,bmargin=F(1,100),tmargin=F(1,20))
            actual=self.m().calibrate(counts,radius)
            if reference is None:self.assertIsNone(actual)
            else:self.assertEqual(F(actual['minimum_reward']),reference[0])

    def test_bad_or_duplicate_rows_are_not_silently_deleted(self):
        m=self.m();rows=self.rows()
        with self.assertRaises(ValueError):m.completion_counts(rows+[rows[0]])
        rows[0]['s']=.5
        with self.assertRaises(ValueError):m.completion_counts(rows)

    def test_policy_bounds_keep_unknown_case_and_distinguish_adjudication_from_truth(self):
        rows=[dict(id='a',task='T',source_hash='x',s=1,a=1,y=1,m_tokens=2,a_tokens=8),
              dict(id='b',task='T',source_hash='y',s=0,a=1,y=0,m_tokens=2,a_tokens=8),
              dict(id='c',task='T',source_hash='z',s=None,a=0,y=1,m_tokens=None,a_tokens=8)]
        got=self.m().policy_bounds(rows,'M:10',R=F(4),D=F(2))
        self.assertEqual(got['accepted'],[1,1]);self.assertEqual(got['queries'],[1,2])
        self.assertEqual(got['fn'],[1,1]);self.assertEqual(got['fp'],[0,0])
        self.assertEqual(got['mean_credit'],[-1/3,1/3])
        self.assertTrue(got['conditional_tokens']['unknown_possible'])
        baseline=self.m().policy_bounds(rows,'blind',R=F(4),D=F(2))
        self.assertEqual(baseline['fp'],[1,1]);self.assertEqual(baseline['conditional_tokens']['bounds'],[24,24])


    def test_metadata_only_is_not_m_plus_with_adjudication(self):
        rows=[dict(id='a',task='T',source_hash='x',s=1,a=0,y=1,m_tokens=2),
              dict(id='b',task='T',source_hash='y',s=None,a=1,y=0,m_tokens=None)]
        v=self.m().metadata_only_bounds(rows)
        self.assertEqual(v['fn'],[0,0]);self.assertEqual(v['fp'],[0,1])
        self.assertEqual(v['adjudications'],0);self.assertEqual(v['screenings'],2)
        self.assertEqual(v['conditional_tokens'],[2,None])

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Run with bundled numpy Python')
    def test_cluster_bootstrap_resamples_sources_not_repeated_responses(self):
        rows=[dict(id=f'{s}-{j}',task='T',source_hash=s,s=int(s=='one'),a=int(s=='one'))
              for s in ('zero','one') for j in range(20)]
        out=self.m().bootstrap(rows,replicates=1000,seed=17)
        self.assertEqual(out['sources'],2);self.assertEqual(out['responses'],40)
        self.assertEqual(out['metrics']['h']['interval'],[0.,1.])
        self.assertEqual(out['metrics']['delta']['interval'][0],0.)

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Run with bundled numpy Python')
    def test_bootstrap_keeps_large_missing_envelope_and_undefined_conditionals(self):
        rows=[dict(id=str(i),task='T',source_hash=str(i),s=None,a=0) for i in range(20)]
        out=self.m().bootstrap(rows,replicates=100,seed=19)
        self.assertEqual(out['missing_rows'],20)
        self.assertEqual(out['metrics']['h']['interval'],[0.,1.])
        self.assertIsNone(out['metrics']['a_plus']['interval'])
        self.assertEqual(out['metrics']['a_plus']['undefined_draws'],100)
        self.assertEqual(out['metrics']['delta']['interval'],[0.,0.])

    @unittest.skipUnless(importlib.util.find_spec('numpy'),'Run with bundled numpy Python')
    def test_bootstrap_envelopes_match_literal_two_world_payoffs(self):
        rows=[dict(id='a',task='T',source_hash='only',s=None,a=1),
              dict(id='b',task='T',source_hash='only',s=1,a=0)]
        out=self.m().bootstrap(rows,replicates=251,seed=19,R=F(4),D=F(2))
        want={'h':[.5,1.],'a0':[.5,.5],'a_plus':[0.,.5],'delta':[-.25,0.],
              'target_credit':[-2.,0.],'blind_credit':[1.,1.],'target_menu_margin':[-3.,-1.]}
        for key,value in want.items():self.assertEqual(out['metrics'][key]['interval'],value)
        self.assertIsNone(out['metrics']['a_minus']['interval'])
        self.assertEqual(out['metrics']['a_minus']['undefined_draws'],251)
        self.assertEqual(out['metrics']['weak_min_reward']['infinite_upper_draws'],251)

if __name__=='__main__':unittest.main()
