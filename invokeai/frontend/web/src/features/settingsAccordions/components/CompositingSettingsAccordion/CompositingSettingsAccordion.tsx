import type { FormLabelProps } from '@invoke-ai/ui';
import {
  Flex,
  FormControlGroup,
  StandaloneAccordion,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
} from '@invoke-ai/ui';
import ParamCanvasCoherenceMode from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceMode';
import ParamCanvasCoherenceSteps from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceSteps';
import ParamCanvasCoherenceStrength from 'features/parameters/components/Canvas/Compositing/CoherencePass/ParamCanvasCoherenceStrength';
import ParamMaskBlur from 'features/parameters/components/Canvas/Compositing/MaskAdjustment/ParamMaskBlur';
import ParamMaskBlurMethod from 'features/parameters/components/Canvas/Compositing/MaskAdjustment/ParamMaskBlurMethod';
import ParamInfillMethod from 'features/parameters/components/Canvas/InfillAndScaling/ParamInfillMethod';
import ParamInfillOptions from 'features/parameters/components/Canvas/InfillAndScaling/ParamInfillOptions';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const coherenceLabelProps: FormLabelProps = {
  minW: '4.5rem',
};

export const CompositingSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'compositing-settings',
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion
      isOpen={isOpen}
      onToggle={onToggle}
      label={t('accordions.compositing.title')}
    >
      <Tabs variant="collapse">
        <TabList>
          <Tab>{t('accordions.compositing.coherenceTab')}</Tab>
          <Tab>{t('accordions.compositing.infillTab')}</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <FormControlGroup formLabelProps={coherenceLabelProps}>
                <ParamCanvasCoherenceMode />
                <ParamCanvasCoherenceSteps />
                <ParamCanvasCoherenceStrength />
                <ParamMaskBlurMethod />
                <ParamMaskBlur />
              </FormControlGroup>
            </Flex>
          </TabPanel>
          <TabPanel>
            <Flex gap={4} p={4} flexDir="column">
              <FormControlGroup formLabelProps={coherenceLabelProps}>
                <ParamInfillMethod />
                <ParamInfillOptions />
              </FormControlGroup>
            </Flex>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </StandaloneAccordion>
  );
});

CompositingSettingsAccordion.displayName = 'CompositingSettingsAccordion';
