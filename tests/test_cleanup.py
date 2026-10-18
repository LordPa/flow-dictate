import json
import os

import numpy as np
import tempfile
import unittest
from pathlib import Path

os.environ["FLOW_DICTATE_HOME"] = tempfile.mkdtemp()

from flow_dictate.cleanup import (  # noqa: E402
    rule_based_cleanup, sanity_check, clean_transcript, join_with_previous, assemble_words,
    find_cut, has_speech, looks_like_cleanup,
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

    def test_reply_detection(self):
        self.assertFalse(looks_like_cleanup("can you hear me", "Yes, I can hear you."))
        self.assertFalse(looks_like_cleanup("what time is it", "It is 3pm."))
        self.assertTrue(looks_like_cleanup("um can you hear me", "Can you hear me?"))
        self.assertTrue(looks_like_cleanup("the doctor said get out of here",
                                           'The doctor said, "Get out of here."'))
        self.assertTrue(looks_like_cleanup("send it tuesday no wednesday at three pm",
                                           "Send it Wednesday at 3pm."))
        self.assertTrue(looks_like_cleanup("I want to go their tomorrow", "I want to go there tomorrow."))

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


class PauseTests(unittest.TestCase):
    """'So tell me what you're thinking [pause] do you want a glass of water?'"""

    def words(self, spec):
        # spec: list of (word, start); each word lasts 0.2s
        return [(w, t, t + 0.2) for w, t in spec]

    def test_long_pause_without_punctuation_ends_sentence(self):
        w = self.words([(" So", 0.0), (" tell", 0.3), (" me", 0.6), (" what", 0.9), (" you're", 1.2),
                        (" thinking", 1.5), (" do", 3.0), (" you", 3.3), (" want", 3.6), (" water?", 3.9)])
        self.assertEqual(assemble_words(w, 0.8), "So tell me what you're thinking. Do you want water?")

    def test_short_pause_is_left_alone(self):
        w = self.words([(" I", 0.0), (" think", 0.3), (" we", 0.9), (" should", 1.2)])
        self.assertEqual(assemble_words(w, 0.8), "I think we should")

    def test_existing_punctuation_is_respected(self):
        w = self.words([(" Hello,", 0.0), (" there", 2.0), (" friend.", 2.3), (" bye", 5.0)])
        self.assertEqual(assemble_words(w, 0.8), "Hello, there friend. Bye")

    def test_disabled(self):
        w = self.words([(" one", 0.0), (" two", 5.0)])
        self.assertEqual(assemble_words(w, 0), "one two")

    def test_segment_start_mid_sentence_is_lowercased(self):
        # "If you look at these two things | You can see that." : Whisper split the
        # segment and capitalized "You" although nothing ended the sentence.
        w = [(" If", 0.0, 0.2, True), (" you", 0.3, 0.5, False), (" look", 0.6, 0.8, False),
             (" You", 1.1, 1.3, True), (" can", 1.4, 1.6, False), (" see.", 1.7, 1.9, False),
             (" Sam", 2.1, 2.3, True), (" left.", 2.4, 2.6, False)]
        self.assertEqual(assemble_words(w, 0.8), "If you look you can see. Sam left.")
        w2 = [(" talk", 0.0, 0.2, True), (" to", 0.3, 0.5, False), (" Wispr", 0.7, 0.9, True),
              (" Flow", 1.0, 1.2, False)]
        self.assertEqual(assemble_words(w2, 0.8, ["Wispr Flow"]), "talk to Wispr Flow")
        self.assertEqual(assemble_words(w2, 0.8), "talk to wispr Flow")


class LiveCutTests(unittest.TestCase):
    SR = 16000

    def audio(self, pattern):
        """pattern: list of (kind, seconds); kind 'speech' or 'quiet'."""
        rng = np.random.default_rng(1)
        parts = []
        for kind, sec in pattern:
            n = int(sec * self.SR)
            amp = 0.1 if kind == "speech" else 0.001
            parts.append((rng.standard_normal(n) * amp).astype("float32"))
        return np.concatenate(parts)

    def test_cut_lands_in_the_last_long_pause(self):
        a = self.audio([("speech", 3.0), ("quiet", 0.8), ("speech", 2.0), ("quiet", 0.1)])
        cut = find_cut(a, 0, self.SR)
        self.assertIsNotNone(cut)
        self.assertTrue(3.0 * self.SR <= cut <= 3.8 * self.SR, cut / self.SR)

    def test_no_cut_without_a_pause_or_enough_audio(self):
        self.assertIsNone(find_cut(self.audio([("speech", 4.0)]), 0, self.SR))
        self.assertIsNone(find_cut(self.audio([("speech", 1.0), ("quiet", 1.0)]), 0, self.SR))

    def test_cut_respects_already_committed_audio(self):
        a = self.audio([("speech", 3.0), ("quiet", 0.8), ("speech", 1.0), ("quiet", 0.8), ("speech", 0.5)])
        first = find_cut(a, 0, self.SR)
        self.assertIsNone(find_cut(a, first, self.SR))  # only 1s of new speech after the first cut

    def test_has_speech(self):
        self.assertTrue(has_speech(self.audio([("speech", 0.5)]), self.SR))
        self.assertFalse(has_speech(self.audio([("quiet", 0.5)]), self.SR))


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
