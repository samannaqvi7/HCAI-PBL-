from django.core.management.base import BaseCommand

from project3 import precompute


class Command(BaseCommand):
    help = (
        'Trains/evaluates the project3 baseline, expert simulation, learning-to-defer, '
        'and active-learning pipelines once, then writes all resulting artifacts '
        '(models, metrics.json, plots, demo/expert-session sample pools) to '
        'project3/artifacts/.'
    )

    def add_arguments(self, parser):
        parser.add_argument('--budget', type=int, default=3000,
                             help='Active-learning query budget (default: 3000).')
        parser.add_argument('--batch-size', type=int, default=100,
                             help='Active-learning batch size per round (default: 100).')

    def handle(self, *args, **options):
        self.stdout.write('Building project3 artifacts -- this takes a few minutes...')
        summary = precompute.run(budget=options['budget'], batch_size=options['batch_size'])
        self.stdout.write(self.style.SUCCESS(
            'project3 artifacts built successfully in {elapsed:.1f}s:\n'
            '  baseline test accuracy:        {baseline_test_accuracy:.4f}\n'
            '  expert overall accuracy:       {expert_overall_accuracy:.4f}\n'
            '  system accuracy @ best thresh: {system_accuracy_at_best:.4f} '
            '(threshold={best_threshold:.2f}, deferral_rate={deferral_rate_at_best:.4f})\n'
            '  oracle accuracy:               {oracle_accuracy:.4f}'.format(
                elapsed=summary['elapsed_seconds'],
                baseline_test_accuracy=summary['baseline_test_accuracy'],
                expert_overall_accuracy=summary['expert_overall_accuracy'],
                system_accuracy_at_best=summary['system_accuracy_at_best'],
                best_threshold=summary['best_threshold'],
                deferral_rate_at_best=summary['deferral_rate_at_best'],
                oracle_accuracy=summary['oracle_accuracy'],
            )
        ))
