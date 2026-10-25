"""Exact reviewed selected/phase/skip/JUnit join; no pytest execution."""

def join_unit_junit(unit, suites, required):
    """Bijection of actual selected/runtime and collection skips to real XML."""
    items = unit['selected']
    assert len({x['nodeid'] for x in items}) == len(items)
    assert len({x['pytest_nodeid'] for x in items}) == len(items)
    assert set(unit['reports']) == {x['pytest_nodeid'] for x in items}
    expected, runtime_skips, collection_skips = {}, [], []
    for item in items:
        address = item['junit_address']
        key = (address['classname'], address['name'])
        assert key not in expected, 'Duplicate actual JUnit address: ' + repr(key)
        phases = unit['reports'][item['pytest_nodeid']]
        order = [x['when'] for x in phases]
        assert order in (['setup', 'call', 'teardown'], ['setup', 'teardown'])
        assert phases[-1]['outcome'] == 'passed' and not phases[-1]['xfail']
        assert all(x['outcome'] in {'passed', 'skipped'} for x in phases)
        skipped = [x for x in phases if x['outcome'] == 'skipped']
        assert len(skipped) <= 1
        if skipped:
            skip = skipped[0]
            assert (skip['when'] == 'setup' and order == ['setup', 'teardown']) or (
                skip['when'] == 'call' and order == ['setup', 'call', 'teardown'] and phases[0]['outcome'] == 'passed')
            assert skip['junit_skip'] is not None and not skip['junit_skip']['truncated']
            runtime_skips.append({'nodeid': item['nodeid'], 'pytest_nodeid': item['pytest_nodeid'],
                'params': item['params'], 'when': skip['when'], 'xfail': skip['xfail'],
                'longrepr': skip['longrepr'], 'junit_skip': skip['junit_skip']})
        else:
            assert order == ['setup', 'call', 'teardown']
            assert all(x['outcome'] == 'passed' and not x['xfail'] and x['junit_skip'] is None for x in phases)
        if item['nodeid'] in required:
            assert not skipped and order == ['setup', 'call', 'teardown']
            assert all(x['outcome'] == 'passed' and not x['xfail'] for x in phases)
        expected[key] = skipped[0]['junit_skip'] if skipped else None
    for report in unit['collection_reports']:
        assert report['outcome'] == 'skipped', 'Failed collection report: ' + report['nodeid']
        address = report['junit_address']
        key = (address['classname'], address['name'])
        assert key not in expected, 'Duplicate/colliding collection JUnit address: ' + repr(key)
        assert report['junit_skip'] is not None and not report['junit_skip']['truncated']
        expected[key] = report['junit_skip']
        collection_skips.append(report)
    cases = list(suites.iter('testcase'))
    actual = {}
    assert cases and not list(suites.iter('failure')) and not list(suites.iter('error'))
    for case in cases:
        key = (case.attrib['classname'], case.attrib['name'])
        assert key not in actual, 'Duplicate XML testcase: ' + repr(key)
        actual[key] = case
    assert set(actual) == set(expected), 'Missing/extra/parameter-mismatched JUnit addresses'
    for key, expectation in expected.items():
        skipped = actual[key].findall('skipped')
        if expectation is None:
            assert not skipped, 'Unexpected XML skip: ' + repr(key)
        else:
            assert len(skipped) == 1
            assert skipped[0].attrib == expectation['attributes']
            assert (skipped[0].text or '') == expectation['text'], 'Changed XML skip reason: ' + repr(key)
    return {'unit_JUnit_cases': len(cases), 'unit_JUnit_skips': len(runtime_skips) + len(collection_skips),
        'unit_runtime_skips': runtime_skips, 'unit_collection_skips': collection_skips,
        'unit_JUnit_runtime_selected_bijection': True,
        'unit_JUnit_collection_skip_bijection': True}
