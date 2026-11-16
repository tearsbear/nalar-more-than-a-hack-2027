from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from nalar import __main__ as cli
from nalar.config import Config, Layout
from nalar.data import digest


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        folder = self.data / 'tactical'
        folder.mkdir(parents=True)
        stem = folder / 'BAR-ATH_tactical'
        meta = '<root><Team id="h"><Name>Barcelona</Name></Team><Team id="a"><Name>Atletic Club</Name></Team>'
        meta += ''.join(f'<Player id="{i}" teamId="{"h" if i <= 23 else "a"}"/>' for i in range(1, 47)) + '</root>'
        Path(str(stem) + '_atd.xml').write_text(meta)
        players = ';'.join(['0.1,0.4,1'] * 23 + ['0.8,0.4,1'] * 23)
        raw = ''.join(f'{i}:{players}:0.105,0.4,0\n' for i in range(1, 121))
        Path(str(stem) + '_atd.txt').write_text(raw)
        tags = '<root><instance><start>0</start><end>4.8</end><code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label></instance></root>'
        Path(str(stem) + '_smart_tagging.xml').write_text(tags)
        self.config = self.root / 'params.json'
        self.config.write_text(json.dumps(asdict(Config())))
        self.layout = Layout(120, 25, (105.3, 68), ((10, 50), (80, 110)), (10, 90))

    def call(self, output, manifest):
        with patch.object(cli, 'official_layout', return_value=self.layout), redirect_stdout(io.StringIO()):
            return cli.main(['generate', '--data-dir', str(self.data), '--config', str(self.config),
                             '--output', str(output), '--manifest', str(manifest)])

    def test_end_to_end_determinism_and_exact_csv_contract(self):
        output, manifest = self.root / 'NALAR.csv', self.root / 'run.json'
        self.assertEqual(self.call(output, manifest), 0)
        first = output.read_bytes()
        self.assertNotIn(b'\r', first)
        lines = first.decode().splitlines()
        self.assertEqual(lines[0], 'frame,label')
        self.assertEqual(len(lines), 121)
        self.assertEqual([int(line.split(',')[0]) for line in lines[1:]], list(range(120)))
        self.assertEqual(lines[81:91], [f'{i},D' for i in range(80, 90)])
        self.assertEqual(self.call(output, manifest), 0)
        self.assertEqual(output.read_bytes(), first)
        m = json.loads(manifest.read_text())
        self.assertEqual(m['output_sha256'], digest(output))
        self.assertFalse(m['ground_truth_used'])
        self.assertEqual(m['diagnostics']['graded_frames'], 70)

    def test_subprocess_generation_with_GT_access_blocked(self):
        script = '''
import sys
from pathlib import Path
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = str(args[0]).replace('\\\\', '/')
        if '/gt/' in name or 'possession_gt' in name:
            raise RuntimeError('ground truth access prohibited')
sys.addaudithook(guard)
from nalar import __main__ as cli
from nalar.config import Layout
cli.official_layout = lambda: Layout(120,25,(105.3,68),((10,50),(80,110)),(10,90))
raise SystemExit(cli.main(sys.argv[1:]))
'''
        result = subprocess.run([sys.executable, '-c', script, 'generate', '--data-dir', str(self.data),
                                 '--config', str(self.config), '--output', str(self.root / 'out.csv'),
                                 '--manifest', str(self.root / 'run.json')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_input_and_baseline_overwrite_rejected_before_loading(self):
        baseline = cli.ROOT / 'evaluation/results/baseline_smooth50.csv'
        checksum = digest(baseline)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(self.call(baseline, self.root / 'run.json'), 1)
            self.assertEqual(self.call(self.data / 'tactical/accidental.csv', self.root / 'run.json'), 1)
            self.assertEqual(self.call(self.config, self.root / 'run.json'), 1)
            self.assertEqual(self.call(self.root / 'same', self.root / 'same'), 1)
        self.assertEqual(digest(baseline), checksum)

    def test_invalid_run_does_not_replace_existing_output(self):
        output = self.root / 'NALAR.csv'
        output.write_bytes(b'existing output\n')
        self.config.write_text('{}')
        with redirect_stderr(io.StringIO()):
            self.assertEqual(self.call(output, self.root / 'run.json'), 1)
        self.assertEqual(output.read_bytes(), b'existing output\n')

    def test_missing_input_has_sanitized_failure(self):
        for path in (self.data / 'tactical').glob('*atd.txt'):
            path.unlink()
        error = io.StringIO()
        with redirect_stderr(error):
            self.assertEqual(self.call(self.root / 'NALAR.csv', self.root / 'run.json'), 1)
        self.assertIn('missing tactical input asset', error.getvalue())
        self.assertFalse((self.root / 'NALAR.csv').exists())
