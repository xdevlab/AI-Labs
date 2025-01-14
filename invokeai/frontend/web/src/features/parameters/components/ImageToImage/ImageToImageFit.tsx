import { FormControl, FormLabel, Switch } from '@invoke-ai/ui';
import type { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldFitToWidthHeight } from 'features/parameters/store/generationSlice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const ImageToImageFit = () => {
  const dispatch = useAppDispatch();

  const shouldFitToWidthHeight = useAppSelector(
    (state: RootState) => state.generation.shouldFitToWidthHeight
  );

  const handleChangeFit = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      dispatch(setShouldFitToWidthHeight(e.target.checked));
    },
    [dispatch]
  );

  const { t } = useTranslation();

  return (
    <FormControl w="full">
      <FormLabel flexGrow={1}>{t('parameters.imageFit')}</FormLabel>
      <Switch isChecked={shouldFitToWidthHeight} onChange={handleChangeFit} />
    </FormControl>
  );
};

export default memo(ImageToImageFit);
