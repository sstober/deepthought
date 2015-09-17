__author__ = 'sstober'

def extract_performance_score(
        model,
        return_channel,
        selection_channel=None, selection_mode='min', tie_mode='last',
        include_report=True,
        report_channels=None, report_mode=None
):

    import numpy as np

    def get_best(values, mode='min'):
        if mode == 'min':
            best_value = np.min(values)
        elif mode == 'max':
            best_value = np.max(values)
        elif mode == 'last':
            best_value = values[-1]
        else:
            raise RuntimeError('mode must be min, max or last')

        return float(best_value), np.where(values == best_value)[0]

    channels = model.monitor.channels

    if selection_channel is None:
        selection_channel = return_channel
    best_value, best_epochs = get_best(channels[selection_channel].val_record, mode=selection_mode)

    if tie_mode == 'last':
        best_epoch = np.max(best_epochs)
    elif tie_mode == 'first':
        best_epoch = np.min(best_epochs)
    else:
        raise RuntimeError('tie_mode must be first or last')

    return_value = float(channels[return_channel].val_record[best_epoch])

    if not include_report:
        return return_value

    report = '{:>30}[{}] = {:.4f} selected'.format(selection_channel, best_epoch, best_value)

    if report_channels is not None:
        if report_mode is None:
            report_mode = selection_mode

        for ch in report_channels:
            if not ch in channels: continue

            v = float(channels[ch].val_record[best_epoch])
            bv, be = get_best(channels[ch].val_record, mode=report_mode)
            report += '\n{:>30}[{}] = {:.4f}, best={:.4f} in {}'.format(ch, best_epoch, v, bv, be)

    report += '\n{:>30}[{}] = {:.4f} returned'.format(return_channel, best_epoch, return_value)
    return return_value, report


# convenience plot functions

def plot_learning_curve(model, figsize=(15,5)):

    import matplotlib.pyplot as plt
    import numpy as np

    monitor = model.monitor
    channels = monitor.channels

    plt.figure(figsize=figsize)
    plt.grid()
    if 'train_y_misclass' in channels:
        print 'min', np.min(channels['train_y_misclass'].val_record)
        print 'max', np.max(channels['train_y_misclass'].val_record)
        plt.plot(channels['train_y_misclass'].val_record, label='train')

    if 'valid_y_misclass' in channels:
        print 'valid min', np.min(channels['valid_y_misclass'].val_record)
        print 'valid max', np.max(channels['valid_y_misclass'].val_record)
        plt.plot(channels['valid_y_misclass'].val_record, label='valid')

    if 'test_y_misclass' in channels:
        print 'test min', np.min(channels['test_y_misclass'].val_record)
        print 'test max', np.max(channels['test_y_misclass'].val_record)
        plt.plot(channels['test_y_misclass'].val_record, label='test')

    plt.legend(loc='best')
    plt.show()