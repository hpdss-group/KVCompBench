"""Shared component change-isolation regression checks."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import subprocess
import os
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'benchmark/compression_framework')]
from benchmark_routes import select_routes
from measurement_identity import stage_sources

class SharedComponents(unittest.TestCase):
    def setUp(self):
        (ROOT/'benchmark/output').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'benchmark/output');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for folder in ['compression_framework','compressors']:
            shutil.copytree(ROOT/'benchmark'/folder,self.root/'benchmark'/folder,ignore=shutil.ignore_patterns('__pycache__','.runtime'))
        self.routes={}
        for key,name,device in [('gzip','gzip','cpu'),('zstd','zstd:3','cpu'),('splitzip','splitzip','declared'),('gzip_gpu','gzip','declared'),('zstd_gpu','zstd:3','declared')]:
            self.routes[key]=select_routes(self.root/'benchmark/compressors',execution_device=device,codecs=name)[name]
    def ids(self):return {name:stage_sources(self.root,route,'encode') for name,route in self.routes.items()}
    def change(self,name,old,new):
        p=self.root/'benchmark'/name;s=p.read_text(encoding='utf-8-sig');self.assertIn(old,s);p.write_text(s.replace(old,new),encoding='utf8')
    def test_fingerprints_are_stable_across_process_hash_seeds(self):
        code="import sys,json;from pathlib import Path;sys.path.insert(0,"+repr(str(ROOT/'benchmark/compression_framework'))+");from benchmark_routes import select_routes;from measurement_identity import stage_sources;r=Path("+repr(str(self.root))+");route=select_routes(r/'benchmark/compressors',execution_device='cpu',codecs='gzip')['gzip'];print(json.dumps(stage_sources(r,route,'encode'),sort_keys=True))"
        outputs=[subprocess.check_output([sys.executable,'-B','-c',code],env=dict(os.environ,PYTHONHASHSEED=str(seed)),text=True,timeout=30) for seed in (1,2)]
        self.assertEqual(outputs[0],outputs[1])
    def test_catalog_algorithm_branch_isolation(self):
        before=self.ids();self.change('compression_framework/adapters/catalog.py','level=int(level or 3)','level=int(level or 4)');after=self.ids()
        for name in ['gzip','gzip_gpu','splitzip']:self.assertEqual(before[name],after[name],name)
        self.assertNotEqual(before['zstd'],after['zstd'])
    def test_imported_function_does_not_depend_on_other_class(self):
        before=self.ids();self.change('compression_framework/adapters/nvcomp_benchmark.py','def codec(self,algorithm,raw):','def codec(self,algorithm,raw):\n        marker = 1');after=self.ids()
        self.assertEqual(before['splitzip'],after['splitzip'])
        self.assertNotEqual(before['zstd_gpu'],after['zstd_gpu'])
    def test_shared_packet_only_affects_consumers(self):
        before=self.ids();self.change('compression_framework/adapters/paper_support.py','def packet(header, buffers):','def packet(header, buffers):\n    marker = 1');after=self.ids()
        self.assertEqual(before['gzip'],after['gzip']);self.assertNotEqual(before['splitzip'],after['splitzip'])
    def test_provider_backend_isolation(self):
        before=self.ids();self.change('compressors/nvcomp/frame_backends.py','level=int(options.get("level", 3))','level=int(options.get("level", 4))');after=self.ids()
        self.assertEqual(before['gzip_gpu'],after['gzip_gpu']);self.assertEqual(before['splitzip'],after['splitzip']);self.assertNotEqual(before['zstd_gpu'],after['zstd_gpu'])
    def test_changed_dispatcher_falls_back_conservatively(self):
        self.change('compression_framework/adapters/catalog.py','name=self.NAME',"name='zstd'")
        before=self.ids();self.change('compression_framework/adapters/catalog.py','level=int(level or 3)','level=int(level or 4)');after=self.ids()
        self.assertNotEqual(before['gzip'],after['gzip'])

if __name__=='__main__':unittest.main()
