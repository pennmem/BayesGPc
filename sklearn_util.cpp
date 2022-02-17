#include "sklearn_util.h"


CMatrix* readNpzFile(const string fileName, const string variableName)
{
    cnpy::npz_t f = cnpy::npz_load(fileName.c_str());
    CMatrix* X = new CMatrix(f[variableName].data<double>(), f[variableName].shape[0], f[variableName].shape[1]);
    return X;
}

void getSklearnKernels(CCmpndKern *kern, cnpy::npz_t npz_dict, CMatrix *X, bool structureOnly)
{
  // create covariance function.
    string key;
    string kernel_key;
    string param_key;
    for(auto it=npz_dict.begin(); it!=npz_dict.end(); it++) {
        key = it->first;
        if (key.find_first_of("&") != key.npos) {
            kernel_key = key.substr(key.find_first_of("&")+1);
            param_key = key.substr(0, key.find_first_of("&"));
        }
        else {
            continue;
        }

        CKern *k;
        k = getSklearnKernel(X->getCols(), npz_dict, kernel_key, param_key, structureOnly);
        kern->addKern(k);
    }
}

CKern* getSklearnKernel(unsigned int x_dim, cnpy::npz_t npz_dict, string kernel_key, string param_key, bool structureOnly)
{
    CKern *kern;
    if(kernel_key.compare("lin") == 0) {
        kern = new CLinKern(x_dim);
        if(!structureOnly)
          kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 0);
    }
    else if(kernel_key.compare("RBF") == 0) {
        kern = new CRbfKern(x_dim);
        if(!structureOnly)
        {
            // sklearn encodes length scale rather than inverse width used in CGp
            double length_scale = *npz_dict[param_key + "__length_scale"].data<double>();
            kern->setParam(1.0/(length_scale*length_scale), 0);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 1);
        }
    }
    else if(kernel_key.compare("RationalQuadratic") == 0) {
        kern = new CRatQuadKern(x_dim);
        if(!structureOnly)
        {
            kern->setParam(*npz_dict[param_key + "__alpha"].data<double>(), 0);
            kern->setParam(*npz_dict[param_key + "__length_scale"].data<double>(), 1);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 2);
        }
    }
    else if(kernel_key.compare("DotProduct") == 0) {
        kern = new CPolyKern(x_dim);
        if(!structureOnly)
        {
            ((CPolyKern*)kern)->setDegree(*npz_dict[param_key + "__exponent"].data<double>());
            kern->setParam(1.0, 0);
            double sigma0 = *npz_dict[param_key + "__sigma_0"].data<double>();
            kern->setParam(sigma0*sigma0, 1);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 2);
            cout << "here in poly kern  sigma0 " << sigma0 << endl;
            cout << "poly kern params  " << kern->getParam(0) << " " << kern->getParam(1) << " " << kern->getParam(2) << " " << endl;
            cout << ((CPolyKern*)kern)->getDegree() << endl;
        }
    }
    else if(kernel_key.compare("Matern") == 0) {
        if (!structureOnly) { throw std::invalid_argument("kernel key 'Matern' must be used with sklearn loading. Use 'Matern32' or 'Matern52' to use without sklearn loading."); }
        double nu = *npz_dict[param_key + "__nu"].data<double>();
        if (nu == 1.5) {
            kern = new CMatern32Kern(x_dim);
        }
        else if (nu == 2.5) {
            kern = new CMatern52Kern(x_dim);
        }
        else {
            throw std::invalid_argument("Nu value not implemented for Matern kernel: " + std::to_string(nu));
        }

        if(!structureOnly)
        {
            kern->setParam(*npz_dict[param_key + "__length_scale"].data<double>(), 0);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 1);
        }
    }
    else if(kernel_key.compare("Matern32") == 0) {
        kern = new CMatern32Kern(x_dim);
        if(!structureOnly)
        {
            kern->setParam(*npz_dict[param_key + "__length_scale"].data<double>(), 0);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 1);
        }
    }
    else if(kernel_key.compare("Matern52") == 0) {
        kern = new CMatern52Kern(x_dim);
        if(!structureOnly)
        {
            kern->setParam(*npz_dict[param_key + "__length_scale"].data<double>(), 0);
            kern->setParam(*npz_dict[param_key + "__constant_value"].data<double>(), 1);
        }
    }
    else if(kernel_key.compare("WhiteKernel") == 0) {
        kern = new CWhiteKern(x_dim);
        if(!structureOnly)
        {
            kern->setParam(*(npz_dict[param_key + "__noise_level"].data<double>()), 0);
        }
    }
    else {
        throw std::invalid_argument("Unknown covariance function type: " + kernel_key);
    }
    return kern;
}
