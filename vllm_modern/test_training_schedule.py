import unittest

from training_schedule import TrainingSchedule, resolve_schedule


class ScheduleTests(unittest.TestCase):
    def test_all_candidates_keep_complete_grpo_update_boundaries(self):
        for micro in (2, 4, 8):
            for generation in (32, 64, 128):
                with self.subTest(micro=micro, generation=generation):
                    s = TrainingSchedule(micro, generation)
                    self.assertEqual(s.gradient_accumulation_steps * micro, 128)
                    self.assertEqual(s.rollouts_per_update * generation, 128)
                    self.assertEqual(s.gradient_accumulation_steps % s.steps_per_generation, 0)
                    pubs, gens = s.versions(3, 5)
                    self.assertEqual(pubs, (3, 4, 5))
                    self.assertEqual(gens, (3,) * s.rollouts_per_update + (4,) * s.rollouts_per_update)

    def test_invalid_schedules_fail_instead_of_changing_statistical_batch(self):
        for values in ((3, 32, 128, 8), (4, 48, 128, 8), (4, 256, 128, 8),
                       (4, 32, 128, 3), (False, 32, 128, 8), (4, 32, 128, 1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingSchedule(*values)

    def test_default_matches_frozen_acceptance_schedule(self):
        s = TrainingSchedule()
        self.assertEqual(s.gradient_accumulation_steps, 32)
        self.assertEqual(s.rollouts_per_update, 4)
        self.assertEqual(s.versions(0, 2), ((0, 1, 2), (0,) * 4 + (1,) * 4))

    def selected(self):
        return {"micro_batch_size": 8, "generation_batch_size": 128,
                "answers_per_update": 128, "num_generations": 8,
                "gradient_accumulation_steps": 16, "steps_per_generation": 16,
                "rollouts_per_update": 1}

    def test_formal_defaults_resolve_to_selected_batch(self):
        result = resolve_schedule(self.selected(), require_frozen=True)
        self.assertEqual(result, TrainingSchedule(8, 128))

    def test_formal_override_cannot_silently_restore_old_batch(self):
        with self.assertRaises(ValueError):
            resolve_schedule(self.selected(), micro=4, generation=32, require_frozen=True)

    def test_benchmark_can_explicitly_compare_old_schedule(self):
        self.assertEqual(resolve_schedule(self.selected(), micro=4, generation=32), TrainingSchedule())

    def test_inconsistent_frozen_accumulation_is_rejected(self):
        settings = self.selected()
        settings["gradient_accumulation_steps"] = 32
        with self.assertRaises(ValueError):
            resolve_schedule(settings, require_frozen=True)


if __name__ == "__main__":
    unittest.main()
