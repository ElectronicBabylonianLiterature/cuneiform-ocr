import unittest

from sign_alignment.data_source import SignTextParser


class SignTextParserTest(unittest.TestCase):
    @staticmethod
    def _text_line(*tokens):
        return {'lines': [{'type': 'TextLine', 'content': list(tokens)}]}

    @staticmethod
    def _grapheme(name, enclosure_type=None):
        return {
            'type': 'Grapheme',
            'name': name,
            'enclosureType': enclosure_type or [],
        }

    def test_line_break_splits_text_line(self):
        text_data = self._text_line(
            self._grapheme('A'),
            {'type': 'LineBreak', 'value': '|', 'cleanValue': '|'},
            self._grapheme('B'),
        )

        self.assertEqual(
            SignTextParser.parse_text_lines(text_data),
            [['A'], ['B']],
        )

    def test_multiple_line_breaks_do_not_create_empty_lines(self):
        text_data = self._text_line(
            {'type': 'LineBreak'},
            self._grapheme('A'),
            {'type': 'LineBreak'},
            {'type': 'LineBreak'},
            self._grapheme('B'),
            {'type': 'LineBreak'},
        )

        self.assertEqual(
            SignTextParser.parse_text_lines(text_data),
            [['A'], ['B']],
        )

    def test_compound_grapheme_pipe_is_not_a_line_break(self):
        text_data = self._text_line(
            {'type': 'CompoundGrapheme', 'cleanValue': '|A.B|'},
        )

        self.assertEqual(
            SignTextParser.parse_text_lines(text_data),
            [['|A.B|']],
        )

    def test_broken_sign_filtering_is_unchanged_across_line_break(self):
        text_data = self._text_line(
            self._grapheme('A'),
            self._grapheme('B', ['BROKEN_AWAY']),
            {'type': 'LineBreak'},
            self._grapheme('C'),
        )

        self.assertEqual(
            SignTextParser.parse_text_lines(text_data, filter_broken=True),
            [['A'], ['C']],
        )
        self.assertEqual(
            SignTextParser.parse_text_lines(text_data, filter_broken=False),
            [['A', 'B'], ['C']],
        )


if __name__ == '__main__':
    unittest.main()
