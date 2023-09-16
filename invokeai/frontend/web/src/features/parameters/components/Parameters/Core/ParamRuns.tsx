import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAINumberInput from 'common/components/IAINumberInput';
import IAISlider from 'common/components/IAISlider';
import { setIterations } from 'features/parameters/store/generationSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

const selector = createSelector(
  [stateSelector],
  (state) => {
    const { initial, min, sliderMax, inputMax, fineStep, coarseStep } =
      state.config.sd.iterations;
    const { iterations } = state.generation;
    const { shouldUseSliders } = state.ui;

    const step = state.hotkeys.shift ? fineStep : coarseStep;

    return {
      iterations,
      initial,
      min,
      sliderMax,
      inputMax,
      step,
      shouldUseSliders,
    };
  },
  defaultSelectorOptions
);

const ParamIterations = () => {
  const {
    iterations,
    initial,
    min,
    sliderMax,
    inputMax,
    step,
    shouldUseSliders,
  } = useAppSelector(selector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleChange = useCallback(
    (v: number) => {
      dispatch(setIterations(v));
    },
    [dispatch]
  );

  const handleReset = useCallback(() => {
    dispatch(setIterations(initial));
  }, [dispatch, initial]);

  return (
    <IAINumberInput
      // label={t('parameters.runs')}
      step={step}
      min={min}
      max={inputMax}
      onChange={handleChange}
      value={iterations}
      numberInputFieldProps={{ textAlign: 'center' }}
      formControlProps={{ w: 36 }}
    />
  );
};

export default memo(ParamIterations);
