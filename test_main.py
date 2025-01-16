import unittest
from parameterized import parameterized

from main import are_oct_equivalents


class TestMain(unittest.TestCase):

    @parameterized.expand([

        # different ligands

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'd', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            False
        ],

        # AAAAAA

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1}
            ],
            True
        ],

        # AAAABB

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1}
            ],
            True
        ],

        # ABABCC

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        # ABCCCC

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        # ABABCD

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            False
        ],

    # ABCDEF

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            True
        ],

    # AABBCC

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 1}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 1}

            ],
            True
        ],

    # AABBCD

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            False
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1}

            ],
            True
        ],

    ## AABCDE

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            True
        ],


        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 1, 'is_symmetric': 1}

            ],
            True
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'b', 'is_flipped': 1, 'is_symmetric': 0},
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 1, 'is_symmetric': 0}

            ],
            False
        ],

    ])
    def test_are_oct_equivalents(self, l1, l2, expected):

        self.assertEqual(are_oct_equivalents(l1, l2), expected)

    @parameterized.expand([

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            AssertionError
        ],

        [
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'c', 'is_flipped': 0, 'is_symmetric': 0}
            ],
            [
                {'id': 'a', 'is_flipped': 0, 'is_symmetric': 0},
                {'id': 'b', 'is_flipped': 0, 'is_symmetric': 1},
                {'id': 'd', 'is_flipped': 0, 'is_symmetric': 1}
            ],
            AssertionError
        ],

    ])
    def test_are_oct_equivalents_with_faulty_input(self, l1, l2, expected_error):

        self.assertRaises(expected_error, are_oct_equivalents, l1, l2)

