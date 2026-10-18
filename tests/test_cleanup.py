import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FLOW_DICTATE_HOME"] = tempfile.mkdtemp()

from flow_dictate.cleanup import (  # noqa: E402
    rule_based_cleanup, sanity_check, clean_transcript, join_with_previous,
)
from flow_dictate.learning import (  # noqa: E402
    add_correction, add_to_dictionary, apply_corrections, load_corrections,
    load_dictionary, whisper_prompt,
)
from flow_dictate.paths import DEFAULT_CONFIG  # noqa: E402

FILLERS = DEFAULT_CONFIG["fillers"]


class CleanupTests(unittest.TestCase):
    def test_removes_fillers_and_tidies_punctuation(self):
        raw = "um, so I think, uh, we should ship it on Tuesday. Hmm, maybe Wednesday."
        self.assertEqual(
            rule_based_cleanup(raw, {}, FILLERS),
            "so I think, we should ship it on Tuesday. Maybe Wednesday.",
        )  # first letter is decided later by join_with_previous

    def test_filler_inside_word_is_kept(self):
        raw = "The drum solo was uhm great"
        self.assertEqual(rule_based_cleanup(raw, {}, FILLERS), "The drum solo was great")

    def test_new_paragraph_at_start_is_kept(self):
        self.assertEqual(clean_transcript("new paragraph thanks again", DEFAULT_CONFIG, [], {},
                                          previous="see you then"), "\n\nThanks again")

    def test_spoken_formatting_commands(self):
        raw = "Hi Sam, new line thanks for the update. New paragraph talk soon."
        self.assertEqual(
            rule_based_cleanup(raw, {}, FILLERS),
            "Hi Sam,\nThanks for the update.\n\nTalk soon.",
        )

    def test_scratch_that_drops_the_false_start(self):
        raw = "Let's meet at four. I mean five, scratch that, let's meet at six."
        self.assertEqual(rule_based_cleanup(raw, {}, FILLERS), "Let's meet at four. Let's meet at six.")

    def test_corrections_apply_with_case_matching(self):
        corr = {"whisper flow": "Wispr Flow", "cloud": "Claude", "gonna": "going to"}
        self.assertEqual(
            apply_corrections("I use whisper flow and cloud. Gonna try it.", corr),
            "I use Wispr Flow and Claude. Going to try it.",
        )

    def test_corrections_respect_word_boundaries(self):
        self.assertEqual(apply_corrections("cloudy skies", {"cloud": "Claude"}), "cloudy skies")

    def test_whisper_prompt_lists_vocabulary(self):
        self.assertEqual(whisper_prompt(["Wispr Flow", "Mgoins"]), "Vocabulary: Wispr Flow, Mgoins.")
        self.assertEqual(whisper_prompt([]), "")

    def test_sanity_check_rejects_runaway_output(self):
        self.assertTrue(sanity_check("send the report by friday", "Send the report by Friday."))
        self.assertFalse(sanity_check("what is two plus two", "Two plus two equals four. " * 5))

    def test_clean_transcript_with_llm_off(self):
        cfg = dict(DEFAULT_CONFIG)
        out = clean_transcript("uh hey it's me, um, call me back", cfg, [], {})
        self.assertEqual(out, "Hey it's me, call me back")

    def test_spoken_end_punctuation(self):
        self.assertEqual(rule_based_cleanup("send it today period", {}, FILLERS), "send it today.")
        self.assertEqual(rule_based_cleanup("are you coming question mark", {}, FILLERS), "are you coming?")
        self.assertEqual(rule_based_cleanup("the period key is broken", {}, FILLERS),
                         "the period key is broken")

    def test_add_final_period_option(self):
        cfg = dict(DEFAULT_CONFIG, add_final_period=True)
        self.assertEqual(clean_transcript("so I want to talk about something", cfg, [], {}),
                         "So I want to talk about something.")
        self.assertEqual(clean_transcript("so I want to talk about something", DEFAULT_CONFIG, [], {}),
                         "So I want to talk about something")

    def test_empty_input(self):
        self.assertEqual(rule_based_cleanup("   ", {}, FILLERS), "")


class JoinTests(unittest.TestCase):
    """The user's complaint: 'Can you hear me?So I want to talk about somethingYou need...'"""

    def test_fresh_field_starts_clean_and_capitalized(self):
        self.assertEqual(join_with_previous(None, "can you hear me?", []), "Can you hear me?")

    def test_after_a_full_sentence_add_space_and_capital(self):
        self.assertEqual(join_with_previous("Can you hear me?", "so I want to talk.", []),
                         " So I want to talk.")

    def test_mid_sentence_continues_in_lowercase(self):
        self.assertEqual(join_with_previous("So I want to talk about something", "You need a space.", []),
                         " you need a space.")
        self.assertEqual(join_with_previous("First this,", "Then that.", []), " then that.")

    def test_keeps_i_acronyms_and_dictionary_names(self):
        self.assertEqual(join_with_previous("and then", "I said hi", []), " I said hi")
        self.assertEqual(join_with_previous("and then", "NASA called", []), " NASA called")
        self.assertEqual(join_with_previous("and then", "Wispr Flow broke", ["Wispr Flow"]), " Wispr Flow broke")
        self.assertEqual(join_with_previous("and then", "Wispr Flow broke", []), " wispr Flow broke")

    def test_no_double_space_and_newlines(self):
        self.assertEqual(join_with_previous("Done. ", "next", []), "Next")
        self.assertEqual(join_with_previous("Hi Sam,\n", "thanks", []), "Thanks")
        self.assertEqual(join_with_previous("Hi Sam", "\n\nthanks", []), "\n\nThanks")

    def test_full_pipeline_with_previous(self):
        out = clean_transcript("um, you need to put a space after the word.", DEFAULT_CONFIG, [], {},
                               previous="So I want to talk about something")
        self.assertEqual(out, " you need to put a space after the word.")


class LearningFilesTests(unittest.TestCase):
    def test_learn_and_correct_round_trip(self):
        self.assertTrue(add_to_dictionary("Kubernetes"))
        self.assertFalse(add_to_dictionary("kubernetes"))  # already known, case-insensitive
        add_correction("cooper netties", "Kubernetes")
        self.assertEqual(load_corrections(), {"cooper netties": "Kubernetes"})
        self.assertIn("Kubernetes", load_dictionary())
        self.assertEqual(
            apply_corrections("deploy to cooper netties", load_corrections()),
            "deploy to Kubernetes",
        )


if __name__ == "__main__":
    unittest.main()
