import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["FLOW_DICTATE_HOME"] = tempfile.mkdtemp()

from flow_dictate.cleanup import rule_based_cleanup, sanity_check, clean_transcript  # noqa: E402
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
            "So I think, we should ship it on Tuesday. Maybe Wednesday.",
        )

    def test_filler_inside_word_is_kept(self):
        raw = "The drum solo was uhm great"
        self.assertEqual(rule_based_cleanup(raw, {}, FILLERS), "The drum solo was great")

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

    def test_empty_input(self):
        self.assertEqual(rule_based_cleanup("   ", {}, FILLERS), "")


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
