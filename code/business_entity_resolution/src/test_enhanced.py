import unittest
import numpy as np
import enhanced_pipeline as e

class ContextStub:
    def count(self,*args):return 1
    def weight(self,*args):return 2.

class EnhancedTests(unittest.TestCase):
    def test_legal_punctuation_and_unseen_country(self):
        a=('S1-a','Acme LLC','123 Park Street','NeverSeen')
        b=('S2-b','Acme L.L.C.','123 Park St','NeverSeen')
        self.assertEqual(e.extended(a)[1],e.extended(b)[1])
        f=e.features(a,b,[],ContextStub())
        self.assertEqual(len(f),120)
        self.assertTrue(np.isfinite(f).all())

    def test_competitor_context(self):
        a=('S1-a','Acme Plumbing','123 Park Street','NeverSeen')
        b=('S2-b','Other Plumbing','123 Park Street','NeverSeen')
        other=('S1-c','Other Plumbing','123 Park Street','NeverSeen')
        empty=e.features(a,b,[],ContextStub());comp=e.features(a,b,[other],ContextStub())
        self.assertGreater(comp[-7],empty[-7])
        self.assertLess(comp[-4],0)

if __name__=='__main__':unittest.main()
