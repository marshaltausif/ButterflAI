// lib/huggingface.js - REAL HuggingFace integration
export async function searchHFDatasets(query, limit = 10) {
  try {
    const response = await fetch(
      `https://huggingface.co/api/datasets?search=${encodeURIComponent(query)}&limit=${limit}`
    )
    
    if (!response.ok) {
      throw new Error('Failed to search HuggingFace datasets')
    }
    
    const datasets = await response.json()
    
    return datasets.map(ds => ({
      id: ds.id,
      name: ds.id.split('/').pop(),
      source: 'hf',
      size_gb: ds.downloads ? ds.downloads / (1024 * 1024 * 1024) : 0,
      num_images: ds.cardData?.size?.split(' ')?.[0] ? 
        parseInt(ds.cardData.size.split(' ')[0]) * 1000 : 0,
      num_classes: ds.cardData?.num_classes || 0,
      relevance_pct: ds.likes ? Math.min(100, ds.likes * 5) : 50,
      license: ds.cardData?.license || 'Unknown',
      description: ds.description || ds.cardData?.description || '',
      format: ds.tags?.includes('image') ? 'datasets' : 'unknown',
      has_train_val_split: ds.cardData?.splits?.length > 0,
      download_cmd: null,
      url: `https://huggingface.co/datasets/${ds.id}`,
      tags: ds.tags || []
    }))
  } catch (error) {
    console.error('HF search failed:', error)
    throw error
  }
}

export async function getHFDatasetInfo(datasetId) {
  try {
    const response = await fetch(`https://huggingface.co/api/datasets/${datasetId}`)
    
    if (!response.ok) {
      throw new Error('Failed to fetch dataset info')
    }
    
    return await response.json()
  } catch (error) {
    console.error('HF info fetch failed:', error)
    throw error
  }
}