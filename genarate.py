from utils.argparser import args
from time import time
from rdkit import Chem
from utils import environment as env
from utils.data_utils import construct_mol
import numpy as np
import torch.nn.functional as F
import torch


def sample_z(model, batch_size=args.batch_size, z_mu=None):
    z_dim = model.adj_size + model.x_size
    mu = np.zeros([z_dim], dtype=np.float32)
    sigma_diag = np.ones([z_dim])
    sigma_diag = np.sqrt(np.exp(model.prior_ln_var.item())) * sigma_diag
    sigma = args.temp * sigma_diag

    if z_mu is not None:
        mu = z_mu
        sigma = 0.01 * np.eye(z_dim, dtype=np.float32)

    z = np.random.normal(mu, sigma, (batch_size, z_dim)).astype(np.float32)
    z = torch.from_numpy(z).float().to(args.device)
    return z.detach()



def generate_one(model, mute=False, cnt=None):
    """
    inverse flow to generate one molecule
    Args:
        temp: temperature of normal distributions, we sample from (0, temp^2 * I)
    """
    generate_start_t = time()
    num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
    num2bond_symbol = {0: '=', 1: '==', 2: '==='}
    num2atom = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}
    num2symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}
    is_continue = True
    mol = None
    total_resample = 0
    batch_size = 1
    # Generating
    z = sample_z(model, batch_size=1)
    A, X = model.reverse(z, model.x_size) # For QM9: [16,9,9,5], [16,9,5], [16,8]-[B,z_dim]
    X = F.softmax(X, dim=2)
    mols = [construct_mol(x_elem, adj_elem, args.atomic_num_list)
            for x_elem, adj_elem in zip(X, A)]
    pure_valid = 0
    smiles = ''
    num_atoms = -1
    for mol in mols:
        assert mol is not None, 'mol is None...'
        final_valid = env.check_chemical_validity(mol)
        valency_valid = env.check_valency(mol)

        if final_valid is False or valency_valid is False:
            print('Warning: use valency check during generation but the final molecule is invalid!!!')
            continue
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        smiles = Chem.MolToSmiles(mol)

        if total_resample == 0:
            pure_valid = 1.0
        if not mute:
            cnt = str(cnt) if cnt is not None else ''
            print('smiles%s: %s | #atoms: %d | #bonds: %d | #resample: %.5f | time: %.5f |' % (
                cnt, smiles, num_atoms, num_bonds, total_resample, time() - generate_start_t))
    return smiles, A, X, pure_valid, num_atoms

def generate_mols_along_axis(model, z0=None, axis=None, n_mols=20, delta=0.1):
    z_list = []

    if z0 is None:
        z0 = sample_z(model, batch_size=1)
        
    for dx in range(n_mols):
        z = z0 + axis * delta * dx
        z_list.append(z)
        
    z_array = np.array(z_list, dtype=np.float32)
    
    A, X = model.reverse(z_array, model.x_size) # For QM9: [16,9,9,5], [16,9,5], [16,8]-[B,z_dim]
    X = F.softmax(X, dim=2)
    mols = [construct_mol(x_elem, adj_elem, args.atomic_num_list)
            for x_elem, adj_elem in zip(X, A)]
    
    smiles = [Chem.MolToSmiles(mol) for mol in mols]

    return mols, smiles
