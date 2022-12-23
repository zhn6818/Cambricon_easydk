#include "inference.h"
#include <glog/logging.h>
#include <sys/time.h>

BaseInfer::BaseInfer(const std::string &modelPath, const std::string &func_name, const int device_id)
{
    context.SetDeviceId(device_id);
    context.BindDevice();
    modelLoader = std::make_shared<edk::ModelLoader>(modelPath.c_str(),func_name.c_str());
    memOp.SetModel(modelLoader);
    modelInfer.Init(modelLoader,device_id);
    auto& inputShape = modelLoader->InputShape(0);
    auto& outputShape = modelLoader->OutputShape(0);
    net_n = inputShape.N();
    net_c = inputShape.C();
    net_w = inputShape.W();
    net_h = inputShape.H();
    out_n = outputShape.N();
    out_c = outputShape.C();
    out_h = outputShape.H();
    out_w = outputShape.W();
    edk::MluResizeConvertOp::Attr rc_attr;
    rc_attr.dst_h = net_h;
    rc_attr.dst_w = net_w;
    rc_attr.batch_size = net_n;
    rc_attr.core_version = context.GetCoreVersion();
    resizeOp.SetMluQueue(modelInfer.GetMluQueue());
    if(!resizeOp.Init(rc_attr))
    {
        THROW_EXCEPTION(edk::Exception::INTERNAL,resizeOp.GetLastError());
    }
    cpuInput = memOp.AllocCpuInput();
    mluInput = memOp.AllocMluInput();
    cpuOutput = memOp.AllocCpuOutput();
    mluOutput = memOp.AllocMluOutput();
    src_rois = new cncvRect[net_n];
    tmp_rois = new cncvRect[net_n];
    CNRT_SAFECALL(cnrtCreateQueue(&queue));
    CNCV_SAFECALL(cncvCreate(&handle));
    CNCV_SAFECALL(cncvSetQueue(handle, queue));
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_src_input, net_n * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_tmp_input, net_n * sizeof(void *)));
    CNRT_SAFECALL(cnrtMalloc((void **) &mlu_dst_input, net_n * sizeof(void *)));
    cpu_src_imgs = new void *[net_n];
    cpu_src_imgs_buff_size = 100 * sizeof(uint8_t);
    for (uint32_t idx = 0; idx < net_n; ++idx) 
    {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
    }
    cpu_tmp_imgs = new void *[net_n];
    int dst_size = net_w * net_h * 3 * sizeof(uint8_t);
    for (uint32_t idx = 0; idx < net_n; ++idx) 
    {
        CNRT_SAFECALL(cnrtMalloc(&(cpu_tmp_imgs[idx]), dst_size));
    }
    cpu_dst_imgs = new void *[net_n];
    // for (uint32_t idx = 0; idx < net_n; ++idx) 
    // {
    //     CNRT_SAFECALL(cnrtMalloc(&(cpu_dst_imgs[idx]), net_w*net_h*net_c*sizeof(uint8_t)));
    // }
    void* mluInputTmp = mluInput[0];
    for(uint32_t idx = 0; idx < net_n ; idx+=1)
    {
        cpu_dst_imgs[idx] = mluInputTmp + idx * net_c * net_w * net_h;
    }
    CNCV_SAFECALL(cncvGetResizeRgbxWorkspaceSize(net_n, &workspace_size));
    CNRT_SAFECALL(cnrtMalloc(&workspace, workspace_size));
}

BaseInfer::~BaseInfer()
{
    if (cpuInput != nullptr)
        memOp.FreeCpuInput(cpuInput);
    // if (mluInput != nullptr)
    //     memOp.FreeMluInput(mluInput);
    if (cpuOutput != nullptr)
        memOp.FreeCpuOutput(cpuOutput);
    if (mluOutput != nullptr)
        memOp.FreeMluOutput(mluOutput);
    cnrtFree(workspace);
    cncvDestroy(handle);
    cnrtDestroyQueue(queue);
    for (uint32_t i = 0; i < net_n; ++i) {
        cnrtFree(cpu_src_imgs[i]);
        cnrtFree(cpu_tmp_imgs[i]);
    }
    cnrtFree(mlu_src_input);
    cnrtFree(mlu_tmp_input);
    cnrtFree(mlu_dst_input);
    delete[] cpu_src_imgs;
    delete[] cpu_tmp_imgs;
    delete[] cpu_dst_imgs;
    delete[] src_rois;
    delete[] tmp_rois;
}

void BaseInfer::resizeCvtColorCpu(const std::vector<cv::Mat>& imgs)
{
    assert(imgs.size() > 0 && imgs.size() <= net_n);
    uchar* cpuInputTmp = (uchar*)(cpuInput[0]);
    for (int n = 0; n < imgs.size(); n++)
    {
        cv::Mat mat = imgs[n].clone();
        cv::resize(mat, mat, cv::Size(net_w, net_h));
        //cv::cvtColor(mat,mat,cv::COLOR_BGR2RGB);
        //cv::Mat floatMat (net_h,net_w,CV_32FC3,cpuInputTmp);
        cv::Mat mergeMat(net_h, net_w, CV_8UC4, cpuInputTmp);
        std::vector<cv::Mat> vec;
        cv::split(mat, vec);
        std::reverse(vec.begin(), vec.end());
        cv::Mat alpha = cv::Mat(net_h, net_w, CV_8UC1);
        vec.push_back(alpha);
        cv::merge(vec, mergeMat);

        // mat.convertTo(floatMat,CV_32FC3,1.0 / 255.0);
        // cv::cvtColor(floatMat,floatMat,CV_BGR2RGB);
        // cv::Mat mean(floatMat.size(),CV_32FC3,cv::Scalar(0.485, 0.456, 0.406));
        // cv::Mat std(floatMat.size(),CV_32FC3,cv::Scalar(0.229, 0.224, 0.225));
        // cv::subtract(floatMat,mean,floatMat);
        // cv::divide(floatMat,std,floatMat);
        cpuInputTmp += net_h * net_w * net_c;
    }
    // for(int i = 0; i <net_n*net_h*net_w*net_c;i++ )
    // {
    //     ((uchar*)cpuInput[0])[i]=1.0;
    // }
    cnrtMemcpy(mluInput[0], cpuInput[0], net_n * net_h * net_w * net_c, CNRT_MEM_TRANS_DIR_HOST2DEV);

}

void BaseInfer::resizeCvtColorMlu(const std::vector<cv::Mat>& imgs)
{
    // std::cout << "start preprocessing" << std::endl;
    assert(imgs.size() > 0 && imgs.size() <= net_n);
    uint32_t maxSize = 0;
    for (int i = 0; i < imgs.size(); ++i)
    {
        uint32_t tmpSize = imgs[i].rows * imgs[i].step * sizeof(uint8_t);
        if (tmpSize > maxSize) {
            maxSize = tmpSize;
        }
    }
    int src_w = imgs[0].cols;
    int src_h = imgs[0].rows;
    int src_stride = imgs[0].step;
    // std::cout << " src_w: " << src_w << " src_h: " << src_h << " src_stride: " << src_stride << std::endl;

    while (maxSize > cpu_src_imgs_buff_size)
    {
        for (uint32_t i = 0; i < net_n; ++i) {
            cnrtFree(cpu_src_imgs[i]);
            //std::cout << "free histroy mlu memry" << std::endl;
        }
        cpu_src_imgs_buff_size = maxSize + 128;
        for (uint32_t idx = 0; idx < net_n; ++idx)
        {
            CNRT_SAFECALL(cnrtMalloc(&(cpu_src_imgs[idx]), cpu_src_imgs_buff_size));
            //std::cout << "remalloc mlu memory" << std::endl;
        }
    }
    uint32_t src_size;
    // copy src imgs to mlu
    for (uint32_t idx = 0; idx < net_n; ++idx)
    {
        src_size = src_h * src_stride * sizeof(uint8_t);
        CNRT_SAFECALL(cnrtMemcpy(cpu_src_imgs[idx], imgs[idx].data, src_size, CNRT_MEM_TRANS_DIR_HOST2DEV));
    }
    CNRT_SAFECALL(cnrtMemcpy(mlu_src_input, cpu_src_imgs, net_n * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_SAFECALL(cnrtMemcpy(mlu_tmp_input, cpu_tmp_imgs, net_n * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));
    CNRT_SAFECALL(cnrtMemcpy(mlu_dst_input, cpu_dst_imgs, net_n * sizeof(void*), CNRT_MEM_TRANS_DIR_HOST2DEV));


    src_desc.width = src_w;
    src_desc.height = src_h;
    src_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    src_desc.stride[0] = src_stride;
    src_desc.depth = CNCV_DEPTH_8U;

    tmp_desc.width = net_w;
    tmp_desc.height = net_h;
    tmp_desc.pixel_fmt = CNCV_PIX_FMT_BGR;
    tmp_desc.stride[0] = net_w * 3 * sizeof(uint8_t);
    tmp_desc.depth = CNCV_DEPTH_8U;

    for (uint32_t i = 0; i < net_n; ++i) {
        // init dst rect
        tmp_rois[i].x = 0;
        tmp_rois[i].y = 0;
        tmp_rois[i].w = net_w;
        tmp_rois[i].h = net_h;

        // init src rect
        src_rois[i].x = 0;
        src_rois[i].y = 0;
        src_rois[i].w = src_w;
        src_rois[i].h = src_h;
    }
    CNCV_SAFECALL(cncvResizeRgbx(handle, net_n, src_desc, src_rois, mlu_src_input, tmp_desc, tmp_rois, mlu_tmp_input, workspace_size, workspace, CNCV_INTER_BILINEAR));
    cncvRect dst_roi = tmp_rois[0];
    dst_desc = tmp_desc;
    dst_desc.pixel_fmt = CNCV_PIX_FMT_RGBA;
    dst_desc.stride[0] = net_w * net_c * sizeof(uint8_t);
    // std::cout << "async" << std::endl;
    CNCV_SAFECALL(cncvRgbxToRgbx(handle, net_n, tmp_desc, dst_roi, mlu_tmp_input, dst_desc, dst_roi, mlu_dst_input));

    // wait for task finished
    CNRT_SAFECALL(cnrtSyncQueue(queue));
    //CNRT_SAFECALL(cnrtMemcpy(mluInput, mlu_dst_input, net_n * sizeof(void*), CNRT_MEM_TRANS_DIR_DEV2HOST));
}
int BaseInfer::GetBatch()
{
    return net_n;
}
