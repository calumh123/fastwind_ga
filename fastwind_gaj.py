import numpy as np
import GA
import shutil
import os
import functools
import sys
import time
import subprocess
from schwimmbad import MPIPool
from collections import OrderedDict
from pathlib import Path
from utils import go_to


def create_GA_directory(object_dir):
    object_template_dir = './Inputs' / object_dir
    shutil.copytree(object_template_dir, object_dir)
    run_dir = object_dir / 'Run'
    output_dir = object_dir / 'Output'
    run_dir.mkdir()
    output_dir.mkdir()
    shutil.copytree('TEMPLATE', run_dir / 'TEMPLATE')
    shutil.copytree('inicalc', run_dir / 'inicalc')
    return run_dir, output_dir


def read_ini_file(object_name):
    # How are you sure it has to be the first file? or is there just one?
    ini_file = list(object_name.glob('*.ini'))[0]
    lines_dic = OrderedDict()

    # read in entire file to one list
    with ini_file.open() as f:
        data = f.readlines()

    num_lines = int(data[2][:-1])
    for i in range(num_lines):
        line_name, resolution = data[i * 5 + 3][:-1].split(' ')
        lower_bound, upper_bound = [float(x) for x in data[i * 5 + 4][:-1].split(' ')]
        gamma = float(data[i * 5 + 5][:-1])
        norm_w1, norm_y1, norm_w2, norm_y2 = [float(x) for x in data[i * 5 + 6][:-1].split(' ')]

        # what's the point of having line_name as in in both dicts?
        line_dic = {'line_name': line_name, 'resolution': int(resolution),
                    'lower_bound': lower_bound, 'upper_bound': upper_bound,
                    'gamma': gamma, 'norm_w1': norm_w1, 'norm_y1': norm_y1,
                    'norm_w2': norm_w2, 'norm_y2': norm_y2}
        lines_dic[line_name] = line_dic

    param_names = ['teff', 'logg', 'mdot', 'vinf', 'beta', 'He',
                   'micro', 'vrot', 'macro', 'N', 'C', 'O', 'Si', 'P']
    params = GA.Parameters()

    for idx, val in enumerate(param_names):
        lower, upper, step = [float(x) for x in data[idx + num_lines * 5 + 4][:-1].split(' ')]
        param_range = upper - lower

        if param_range != 0:
            log_abs_range = np.log10(abs(param_range))
            sig_digits = int(np.ceil(log_abs_range) - np.floor(np.log10(step)))
            if log_abs_range % 1 == 0:
                sig_digits += 1
            params.add(val, lower, upper, sig_digits)

    metallicity = data[num_lines * 5 + 5 + len(param_names)].split(' ')[0]
    K_mag = data[num_lines * 5 + 6 + len(param_names)].split(' ')[0]

    pop = int(data[num_lines * 5 + 8 + len(param_names)].split(' ')[0][:-1])
    gens = int(data[num_lines * 5 + 9 + len(param_names)].split(' ')[0][:-1])

    return lines_dic, params, metallicity, K_mag, pop, gens


def renormalize_spectra(lines_dic, object_name):
    norm_file = object_name / object_name.with_suffix('.norm')
    wave, flux, err = np.loadtxt(norm_file).T

    for key, val in lines_dic.items():
        # I guess wave is a list
        # does inds have to be a list? -> see wave[inds]...
        inds = [i for i, v in enumerate(wave)
                if v >= val['lower_bound'] and v <= val['upper_bound']]
        m = (val['norm_y2'] - val['norm_y1']) / (val['norm_w2'] - val['norm_w1'])
        b = val['norm_y1'] - m * val['norm_w1'] + 1
        y = m * wave[inds] + b

        # update values in dict
        lines_dic[key]['norm_wave'] = wave[inds]
        lines_dic[key]['norm_flux'] = flux[inds] / y
        lines_dic[key]['norm_err'] = err[inds]

    return lines_dic


def create_model_directory(run_dir, param_set):
    run_id = param_set['run_id']
    model_dir = run_dir / run_id
    run_id_dir = model_dir / run_id

    shutil.copytree(run_dir / 'TEMPLATE', model_dir)
    run_id_dir.mkdir()

    return model_dir


def create_INDAT_file(run_dir, param_set, metallicity):
    # why not have them saved as string from the start?
    teff = str(param_set['teff'])
    logg = str(param_set['logg'])
    radius = '7.2'
    mdot = str(10**param_set['mdot'])
    v_min = '0.1'
    v_inf = str(param_set['vinf'])
    beta = str(param_set['beta'])
    v_trans = '0.1'
    He = str(param_set['He'])
    num_e = '2'
    micro = '15.0'
    C = str(param_set['C'])
    N = str(param_set['N'])
    O = str(param_set['O'])

    run_id = param_set['run_id']
    indat_file = run_dir / run_id / 'INDAT.DAT'

    with indat_file.open('w') as f:
        f.write(f'\'{run_id}\'\n')
        f.write(' T T           0         100\n')
        f.write('  0.000000000000000E+000\n')
        f.write('   '.join([teff, logg, radius]) + '\n')
        f.write('   120.000000000000       0.600000000000000\n')
        f.write('   '.join([mdot, v_min, v_inf, beta, v_trans]) + '\n')
        f.write('   '.join([He, num_e]) + '\n')
        f.write(' F T F T T\n')
        f.write('   '.join([micro, metallicity]) + ' T T\n')
        f.write(' T F           1           2\n')
        f.write(' 1.000       0.1 0.2\n')
        f.write('\n'.join([f'C    {C}', f'N    {N}', f'O    {O}']) + '\n')


def run_fastwind(run_dir, output_dir, lines_dic, metallicity, param_set):
    model_dir = create_model_directory(run_dir, param_set)
    create_INDAT_file(run_dir, param_set, metallicity)

    run_id = param_set['run_id']
    run_id_1st_part = run_id.split('_')[0]
    run_id_dir = output_dir / run_id_1st_part / run_id
    run_id_dir.mkdir()
    model_run_id = model_dir / run_id
    lines = model_run_id.glob('OUT.*')
    param_list_return = list(param_set.values())

    total_chi2 = 0
    total_deg_of_freedom = 0
    line_fitnesses = []

    with go_to(model_dir):
        # why the try/except?
        try:
            subprocess.run('timeout 1h ./pnlte_A10HHeNCOPSi.eo > temp.txt',
                           shell=True, check=True)
            np.savetxt('temp1.txt', np.array([run_id, '15.0 0.1', '0']), fmt='%s')
            subprocess.run('./pformalsol_A10HHeNCOPSi.eo < temp1.txt > temp2.txt',
                           shell=True, check=True)

            #.system('timeout 1h ./pnlte_A10HHeNCOPSi.eo > temp.txt')
            # os.system('timeout 1h ./pnlte_A10HHeNCOPSi.eo > /dev/null')
            # np.savetxt('temp1.txt', np.array([param_set['run_id'], '15.0 0.1', '0']), fmt='%s')
            # os.system('./pformalsol_A10HHeNCOPSi.eo < temp1.txt > temp2.txt')
            # r = Popen('./pformalsol_A10HHeNCOPSi.eo > temp.txt', stdin=PIPE)
            # r.communicate(os.linesep.join([param_set['run_id'], '15.0 0.1', '0']))
        except:
            pass
    # I guess it went back where you can from
    # os.chdir('../../../')

    if len(list(lines)) <= 1:
        param_list_return.append(999999999)
        param_list_return.append(0.0)
        param_list_return.extend(np.zeros_like(lines_dic.keys()))
        shutil.rmtree(model_dir)
        print(f'failed: {run_id}')

        return param_list_return

    for line in lines:
        # probably able to use line.stem but I don't know how the filename looks
        line_name = line.name.split('.')[-1].split('_')[0]
        x = np.genfromtxt(line, max_rows=161).T
        wavelength = x[2]
        flux = x[4]
        new_line_file_name = model_run_id / line_name / '.prof'
        np.savetxt(new_line_file_name, np.array([wavelength, flux]).T,
                   header='#161     #0', comments='')

    for line, val in lines_dic.items():
        new_line_file_name = model_run_id / line / '.prof'
        broad_command = (f"python broaden.py -f {new_line_file_name}"
                         f" -r {val['resolution']} -v {param_set['vrot']} -m -1")
        # unsure in shell has to be True
        subprocess.run(broad_command, shell=True, check=True)
        # double suffix?
        final_new_line_file_name = new_line_file_name / '.fin'
        chi2, dof = calculate_chi2(final_new_line_file_name, val)
        total_chi2 += chi2
        total_deg_of_freedom += dof
        line_fitnesses.append(1. / chi2)
        shutil.copy(final_new_line_file_name, run_id_dir)

    shutil.copy(model_run_id / 'INDAT.DAT', run_id_dir)
    shutil.rmtree(model_dir)

    print(total_chi2, total_deg_of_freedom)
    total_deg_of_freedom -= len(param_set)
    total_chi2 /= (total_deg_of_freedom**2)
    param_list_return.append(total_chi2)
    param_list_return.append(1. / total_chi2)
    param_list_return.extend(line_fitnesses)

    return param_list_return


def dopler_shift(w, rv):
    c = 299792.458
    return w * c / (c - rv)


def calculate_chi2(exp_fname, line_dic):
    w_broad, f_broad = np.loadtxt(exp_fname).T
    w_shifted = dopler_shift(w_broad, line_dic['gamma'])
    observed_wave = line_dic['norm_wave']
    observed_flux = line_dic['norm_flux']
    observed_err = line_dic['norm_err']
    expected_flux = np.interp(observed_wave, w_shifted, f_broad)
    chi2 = np.sum(((observed_flux - expected_flux) / observed_err)**2)
    deg_of_freedom = len(observed_wave)
    return chi2, deg_of_freedom


if __name__ == "__main__":

    with MPIPool() as pool:

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        start_time_prog = time.time()

        object_name = Path('vfts352a_uvALL91')
        print('Creating GA directory...')
        run_dir, output_dir = create_GA_directory(object_name)
        print('Reading ini file...')

        (lines_dic, params,
         metallicity, K_mag,
         population_size,
         number_of_generations) = read_ini_file(object_name)

        lines_dic = renormalize_spectra(lines_dic, object_name)

        # unused...
        # keep_files = False
        # population_size = 59
        # number_of_generations = 500
        mutation_rate = 0.005
        print('Creating chromosome...')
        population_raw = GA.create_chromosome(params, population_size)
        # vs = GA.batch_translate_chromosomes(params, population_raw, 0)

        outfile = output_dir / 'chi2.txt'

        # list of list?
        # also: saved as string so why make array?
        param_list = np.array([[val.name, val.min, val.max, val.precision]
                               for key, val in params.items()])
        np.savetxt(output_dir / 'params.txt', param_list, fmt='%s')

        best_fitness = 0
        best_mods = []

        number_of_lines = len(lines_dic)

        with outfile.open('w') as f:
            f.write(f'#{" ".join(params.keys())} run_id chi2 fitness '
                    f'{" ".join(lines_dic.keys())}')

        for generation in range(number_of_generations):
            gen_start_time = time.time()

            population = GA.batch_translate_chromosomes(params, population_raw, generation)
            print(f'Generation : {generation}')
            generation_dir = output_dir / f'{generation}'.zfill(4)
            generation_dir.mkdir()

            gen_fitnesses = pool.map(functools.partial(
                run_fastwind, run_dir, output_dir, lines_dic, metallicity), population)
            gen_fitnesses_array = np.array(gen_fitnesses)
            with outfile.open('a') as f:
                np.savetxt(f, gen_fitnesses_array, fmt='%s')

            # array of array?
            fitness = np.array(gen_fitnesses_array[:, -1 * number_of_lines - 1],
                               dtype='float')
            print(fitness)
            population_raw = GA.crossover_and_mutate_raw(population_raw, fitness, mutation_rate)
            mutation_rate = GA.adjust_mutation_rate(mutation_rate, fitness)

            if np.max(fitness) > best_fitness:
                best_fitness = np.max(fitness)
                best_mod = population[np.argmax(fitness)]
            best_mods.append(best_mod)

            gen_time = time.time()
            print('Since start: ' + str(gen_time - start_time_prog))
            print('Gen time: ' + str(gen_time - gen_start_time))

    print(best_fitness)
    print(best_mod)
    exit()
